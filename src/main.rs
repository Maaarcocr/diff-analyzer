//! This is a translation of embedding.cpp in llama.cpp using llama-cpp-2.
#![allow(
    clippy::cast_possible_wrap,
    clippy::cast_possible_truncation,
    clippy::cast_precision_loss,
    clippy::cast_sign_loss
)]

use std::path::PathBuf;
use std::vec;

use anyhow::{Context, Result};
use clap::Parser;
use hf_hub::api::sync::ApiBuilder;

use linfa::Dataset;
use linfa_clustering::KMeans;
use linfa::traits::{Fit, Predict, Transformer};
use llama_cpp_2::context::params::LlamaContextParams;
use llama_cpp_2::context::LlamaContext;
use llama_cpp_2::llama_backend::LlamaBackend;
use llama_cpp_2::llama_batch::LlamaBatch;
use llama_cpp_2::model::params::LlamaModelParams;
use llama_cpp_2::model::LlamaModel;
use llama_cpp_2::model::AddBos;
use ndarray::{Array1, Array2};

#[derive(clap::Parser, Debug, Clone)]
struct Args {
    /// The file that contains the diffs issues to analyse
    filename: String,
    /// Whether to normalise the produced embeddings
    #[clap(short)]
    normalise: bool,
    /// the repo containing the model. e.g. `BAAI/bge-small-en-v1.5`
    #[clap(default_value = "lm-kit/bge-m3-gguf", long)]
    repo: String,
    /// the model name. e.g. `BAAI-bge-small-v1.5.Q4_K_M.gguf`
    #[clap(default_value = "bge-m3-Q8_0.gguf", long)]
    model: String,
    /// number of clusters
    #[clap(short, long, default_value = "8")]
    clusters: usize,
}

fn get_or_load(model: String, repo: String) -> Result<PathBuf> {
    ApiBuilder::new()
        .with_progress(true)
        .build()
        .with_context(|| "unable to create huggingface api")?
        .model(repo)
        .get(&model)
        .with_context(|| "unable to download model")
}

fn main() -> Result<()> {
    let Args {
        model,
        filename,
        normalise,
        repo,
        clusters
    } = Args::parse();

    // init LLM
    let mut backend = LlamaBackend::init()?;
    backend.void_logs();

    // offload all layers to the gpu
    let model_params = LlamaModelParams::default();

    let model_path = get_or_load(model, repo)
        .with_context(|| "failed to get model from args")?;

    let model = LlamaModel::load_from_file(&backend, model_path, &model_params)
        .with_context(|| "unable to load model")?;

    // initialize the context
    let ctx_params = LlamaContextParams::default()
        .with_n_threads_batch(std::thread::available_parallelism()?.get().try_into()?)
        .with_embeddings(true);

    let mut ctx = model
        .new_context(&backend, ctx_params)
        .with_context(|| "unable to create the llama_context")?;

    let mut rdr = csv::Reader::from_path(&filename).with_context(|| "failed to read CSV file")?;
    let file_lines = rdr
        .records()
        .map(|result| {
            let record = result.with_context(|| "failed to read CSV record")?;
            record
                .get(2) // Assuming "diff" is the third column
                .with_context(|| "failed to get 'diff' column")
                .map(|s| s.to_string())
        })
        .collect::<Result<Vec<_>, _>>()?;

    // tokenize the prompt
    let tokens_lines_list = file_lines.iter()
        .map(|line| model.str_to_token(&line, AddBos::Always))
        .collect::<Result<Vec<_>, _>>()
        .with_context(|| format!("failed to tokenize {filename}"))?;

    let n_ctx = ctx.n_ctx() as usize;

    let tokens_lines_list = tokens_lines_list
        .into_iter()
        .map(|mut tok| {
            if tok.len() > n_ctx {
                tok.truncate(n_ctx);
            }
            tok
        })
        .collect::<Vec<_>>();

    // create a llama_batch with the size of the context
    // we use this object to submit token data for decoding
    let mut batch = LlamaBatch::new(n_ctx, 1);

    let mut max_seq_id_batch = 0;
    let mut output = Vec::with_capacity(tokens_lines_list.len());

    for tokens in &tokens_lines_list {
        // Flush the batch if the next prompt would exceed our batch size
        if (batch.n_tokens() as usize + tokens.len()) > n_ctx {
            batch_decode(
                &mut ctx,
                &mut batch,
                max_seq_id_batch,
                &mut output,
                normalise,
            )?;
            max_seq_id_batch = 0;
        }

        batch.add_sequence(tokens, max_seq_id_batch, false)?;
        max_seq_id_batch += 1;
    }
    // Handle final batch
    batch_decode(
        &mut ctx,
        &mut batch,
        max_seq_id_batch,
        &mut output,
        normalise,
    )?;

    let records: Array2<f32> = ndarray::Array2::from_shape_vec(
        (output.len(), output[0].len()),
        output.into_iter().flatten().collect(),
    )?;
    let dataset = Dataset::new(records, Array1::<f32>::zeros(0));
    let model = KMeans::params(clusters).tolerance(1e-4).fit(&dataset)?;
    
    let predicted_clusters = model.predict(&dataset);
    let distances_from_centroid = model.transform(&dataset.records);

    let mut clostest_entry_for_each_cluster = vec![(None, f32::MAX); 8];

    for ((cluster, distance), line) in predicted_clusters.iter().zip(distances_from_centroid.iter()).zip(file_lines.iter()) {
        let (_, closest_distance) = &clostest_entry_for_each_cluster[*cluster as usize];
        if distance < closest_distance {
            clostest_entry_for_each_cluster[*cluster as usize] = (Some(line), *distance);
        }
    }

    for (i, (closest_entry, distance)) in clostest_entry_for_each_cluster.iter().enumerate() {
        if let Some(entry) = closest_entry {
            println!("Cluster {} - Distance: {:.2} - Entry:", i, distance);
            println!("{}", entry);
            println!("---------------------------------");
        }
    }

    Ok(())
    
}

fn batch_decode(
    ctx: &mut LlamaContext,
    batch: &mut LlamaBatch,
    s_batch: i32,
    output: &mut Vec<Vec<f32>>,
    normalise: bool,
) -> Result<()> {
    ctx.clear_kv_cache();
    ctx.decode(batch).with_context(|| "llama_decode() failed")?;

    for i in 0..s_batch {
        let embedding = ctx
            .embeddings_seq_ith(i)
            .with_context(|| "Failed to get embeddings")?;
        let output_embeddings = if normalise {
            normalize(embedding)
        } else {
            embedding.to_vec()
        };

        output.push(output_embeddings);
    }

    batch.clear();

    Ok(())
}

fn normalize(input: &[f32]) -> Vec<f32> {
    let magnitude = input
        .iter()
        .fold(0.0, |acc, &val| val.mul_add(val, acc))
        .sqrt();

    input.iter().map(|&val| val / magnitude).collect()
}