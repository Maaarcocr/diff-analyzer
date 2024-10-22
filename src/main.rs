//! This is a translation of embedding.cpp in llama.cpp using llama-cpp-2.
#![allow(
    clippy::cast_possible_wrap,
    clippy::cast_possible_truncation,
    clippy::cast_precision_loss,
    clippy::cast_sign_loss
)]

use std::io::BufRead;
use std::io::Write;
use std::path::PathBuf;
use std::vec;

use anyhow::{Context, Result};
use clap::Args;
use clap::Parser;
use clap::Subcommand;
use hf_hub::api::sync::ApiBuilder;

use indicatif::ProgressBar;
use indicatif::ProgressStyle;
use linfa::traits::{Fit, Predict, Transformer};
use linfa::Dataset;
use linfa_clustering::KMeans;
use llama_cpp_2::context::params::LlamaContextParams;
use llama_cpp_2::context::LlamaContext;
use llama_cpp_2::llama_backend::LlamaBackend;
use llama_cpp_2::llama_batch::LlamaBatch;
use llama_cpp_2::model::params::LlamaModelParams;
use llama_cpp_2::model::AddBos;
use llama_cpp_2::model::LlamaModel;
use llama_cpp_2::token::LlamaToken;
use llama_cpp_2::StringToTokenError;
use ndarray::{Array1, Array2};
use rayon::iter::IntoParallelRefIterator;
use rayon::iter::ParallelIterator;

#[derive(Debug, Clone, Args)]
struct KMeansArgs {
    #[clap(flatten)]
    common: CommonArgs,
    #[clap(short, long, default_value = "8")]
    clusters: usize,
}

#[derive(Debug, Clone, Args)]
struct TSNEArgs {
    #[clap(flatten)]
    common: CommonArgs,
}

#[derive(Subcommand, Debug, Clone)]
enum Commands {
    KMeans(KMeansArgs),
    TSNE(TSNEArgs),
}

impl Commands {
    fn run(&self, data: Vec<Vec<f32>>, strings: Vec<String>) -> Result<()> {
        match self {
            Commands::KMeans(kmeans_args) => self.kmeans(data, strings, kmeans_args.clusters),
            Commands::TSNE(_) => self.tsne(data, strings),
        }
    }

    fn get_common_args(&self) -> &CommonArgs {
        match self {
            Commands::KMeans(args) => &args.common,
            Commands::TSNE(args) => &args.common,
        }
    }

    fn kmeans(&self, data: Vec<Vec<f32>>, strings: Vec<String>, clusters: usize) -> Result<()> {
        let records: Array2<f32> = ndarray::Array2::from_shape_vec(
            (data.len(), data[0].len()),
            data.into_iter().flatten().collect(),
        )?;
        let dataset = Dataset::new(records, Array1::<f32>::zeros(0));
        let model = KMeans::params(clusters).tolerance(1e-4).fit(&dataset)?;

        let predicted_clusters = model.predict(&dataset);
        let distances_from_centroid = model.transform(&dataset.records);

        let mut clostest_entry_for_each_cluster = vec![(None, f32::MAX); 8];

        for ((cluster, distance), line) in predicted_clusters
            .iter()
            .zip(distances_from_centroid.iter())
            .zip(strings.iter())
        {
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

    fn tsne(&self, data: Vec<Vec<f32>>, _: Vec<String>) -> Result<()> {
        let mut tsne = bhtsne::tSNE::new(&data);
        let embeddings = tsne
            .embedding_dim(2)
            .perplexity(20.0)
            .epochs(200)
            .barnes_hut(0.2, |sample_a, sample_b| {
                sample_a
                    .iter()
                    .zip(sample_b.iter())
                    .map(|(a, b)| (a - b).powi(2))
                    .sum::<f32>()
                    .sqrt()
            });

        let mut file = std::fs::File::create("tsne_visualization.html")
            .with_context(|| "failed to create HTML file")?;

        writeln!(
            file,
            "<!DOCTYPE html>
        <html>
        <head>
            <title>t-SNE Visualization</title>
            <script src=\"https://cdn.jsdelivr.net/npm/chart.js\"></script>
        </head>
        <body>
            <canvas id=\"tsneChart\" width=\"800\" height=\"600\"></canvas>
            <script>
                const ctx = document.getElementById('tsneChart').getContext('2d');
                const data = {{
                    datasets: [{{
                        label: 't-SNE',
                        data: [{}],
                        backgroundColor: 'rgba(75, 192, 192, 0.6)',
                        borderColor: 'rgba(75, 192, 192, 1)',
                        borderWidth: 1,
                        pointRadius: 5,
                    }}]
                }};
                const config = {{
                    type: 'scatter',
                    data: data,
                    options: {{
                        scales: {{
                            x: {{
                                type: 'linear',
                                position: 'bottom'
                            }}
                        }}
                    }}
                }};
                new Chart(ctx, config);
            </script>
        </body>
        </html>",
            embeddings
                .embedding()
                .as_slice()
                .chunks(2)
                .map(|point| format!("{{x: {}, y: {}}}", point[0], point[1]))
                .collect::<Vec<_>>()
                .join(", ")
        )?;

        std::process::Command::new("open")
            .arg("tsne_visualization.html")
            .spawn()
            .with_context(|| "failed to open HTML file")?;

        Ok(())
    }
}

#[derive(Debug, Clone, Args)]
struct CommonArgs {
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
}

#[derive(clap::Parser, Debug, Clone)]
struct CliArgs {
    #[clap(subcommand)]
    subcommand: Commands,
}

fn get_or_load(model: &str, repo: &str) -> Result<PathBuf> {
    ApiBuilder::new()
        .with_progress(true)
        .build()
        .with_context(|| "unable to create huggingface api")?
        .model(repo.to_owned())
        .get(&model)
        .with_context(|| "unable to download model")
}

pub trait Cacheable: Sized {
    fn cache(&self, path: &PathBuf) -> Result<()>;
    fn load(path: &PathBuf) -> Result<Self>;
}

struct Tokens {
    tokens: Vec<Vec<LlamaToken>>,
}

impl Tokens {
    fn new(tokens: Vec<Vec<LlamaToken>>) -> Self {
        Self { tokens }
    }
}

impl Cacheable for Tokens {
    fn cache(&self, path: &PathBuf) -> Result<()> {
        let mut file = std::fs::File::create(path)?;
        for line in &self.tokens {
            for token in line {
                write!(file, "{} ", token.0)?;
            }
            writeln!(file)?;
        }
        Ok(())
    }

    fn load(path: &PathBuf) -> Result<Self> {
        let mut tokens = Vec::new();
        let file = std::fs::File::open(path)?;
        for line in std::io::BufReader::new(file).lines() {
            let line = line?;
            let line = line.trim();
            let line_tokens: Vec<LlamaToken> = line
                .split_whitespace()
                .map(|token| Ok::<_, anyhow::Error>(LlamaToken::new(token.parse()?)))
                .collect::<Result<_>>()?;
            tokens.push(line_tokens);
        }
        Ok(Self { tokens })
    }
}

struct Embeddings {
    embeddings: Vec<Vec<f32>>,
}

impl Embeddings {
    fn new(embeddings: Vec<Vec<f32>>) -> Self {
        Self { embeddings }
    }
}

impl Cacheable for Embeddings {
    fn cache(&self, path: &PathBuf) -> Result<()> {
        let mut file = std::fs::File::create(path)?;
        for line in &self.embeddings {
            for token in line {
                write!(file, "{} ", token)?;
            }
            writeln!(file)?;
        }
        Ok(())
    }

    fn load(path: &PathBuf) -> Result<Self> {
        let mut embeddings = Vec::new();
        let file = std::fs::File::open(path)?;
        for line in std::io::BufReader::new(file).lines() {
            let line = line?;
            let line = line.trim();
            let line_embeddings: Vec<f32> = line
                .split_whitespace()
                .map(|token| Ok::<_, anyhow::Error>(token.parse()?))
                .collect::<Result<_>>()?;
            embeddings.push(line_embeddings);
        }
        Ok(Self { embeddings })
    }
}

fn cache<T: Cacheable, F: FnOnce() -> Result<T>>(path: &PathBuf, f: F) -> Result<T> {
    if path.exists() {
        T::load(path)
    } else {
        let value = f()?;
        value.cache(path)?;
        Ok(value)
    }
}

fn main() -> Result<()> {
    let CliArgs { subcommand } = CliArgs::parse();

    let progress_style = ProgressStyle::default_bar().template("{msg} {wide_bar} {pos}/{len}")?;

    let CommonArgs {
        filename,
        normalise,
        repo,
        model,
    } = subcommand.get_common_args();

    // init LLM
    let mut backend = LlamaBackend::init()?;
    backend.void_logs();

    // offload all layers to the gpu
    let model_params = LlamaModelParams::default();

    let model_path = get_or_load(model, repo).with_context(|| "failed to get model from args")?;

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

    let hash = blake3::hash(&file_lines.join("\n").as_bytes()).to_hex();

    let tokens_cache_path = PathBuf::from(format!("{}.tokens", hash));
    let n_ctx = ctx.n_ctx() as usize;

    let tokens_lines_list = cache(&tokens_cache_path, || {
        let progress_bar = ProgressBar::new(file_lines.len() as u64)
            .with_message("Tokenizing")
            .with_style(progress_style.clone());
        // tokenize the prompt
        let tokens_lines_list = file_lines
            .par_iter()
            .map(|line| {
                let mut tok = model.str_to_token(&line, AddBos::Always)?;
                if tok.len() > n_ctx {
                    tok.truncate(n_ctx);
                }
                progress_bar.inc(1);
                Ok::<_, StringToTokenError>(tok)
            })
            .collect::<Result<Vec<_>, _>>()
            .with_context(|| format!("failed to tokenize {filename}"))?;
        Ok(Tokens::new(tokens_lines_list))
    })?
    .tokens;

    let embeddings_cache_path = PathBuf::from(format!("{}.embeddings", hash));

    let embeddings = cache(&embeddings_cache_path, || {
        // create a llama_batch with the size of the context
        // we use this object to submit token data for decoding
        let mut batch = LlamaBatch::new(n_ctx, 1);

        let mut max_seq_id_batch = 0;
        let mut output = Vec::with_capacity(tokens_lines_list.len());
        let progress_bar = ProgressBar::new(tokens_lines_list.len() as u64)
            .with_message("Embedding")
            .with_style(progress_style);

        for tokens in &tokens_lines_list {
            // Flush the batch if the next prompt would exceed our batch size
            if (batch.n_tokens() as usize + tokens.len()) > n_ctx {
                batch_decode(
                    &mut ctx,
                    &mut batch,
                    max_seq_id_batch,
                    &mut output,
                )?;
                max_seq_id_batch = 0;
            }

            batch.add_sequence(tokens, max_seq_id_batch, false)?;
            max_seq_id_batch += 1;
            progress_bar.inc(1);
        }
        // Handle final batch
        batch_decode(
            &mut ctx,
            &mut batch,
            max_seq_id_batch,
            &mut output,
        )?;
        progress_bar.finish();
        Ok(Embeddings::new(output))
    })?
    .embeddings;

    let embeddings = if *normalise {
        embeddings.into_iter().map(|e| normalize(&e)).collect()
    } else {
        embeddings
    };

    subcommand.run(embeddings, file_lines)?;

    Ok(())
}

fn batch_decode(
    ctx: &mut LlamaContext,
    batch: &mut LlamaBatch,
    s_batch: i32,
    output: &mut Vec<Vec<f32>>,
) -> Result<()> {
    ctx.clear_kv_cache();
    ctx.decode(batch).with_context(|| "llama_decode() failed")?;

    for i in 0..s_batch {
        let embedding = ctx
            .embeddings_seq_ith(i)
            .with_context(|| "Failed to get embeddings")?;

        output.push(embedding.to_vec());
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
