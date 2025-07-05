import tf from "@tensorflow/tfjs-node";
import fs from "node:fs";
import {getInputShape, loadProcessedData} from "./preprocess.js";
import {normalizeEmbeddings} from "./embeddingUtils.js";

/**
 * Load the trained model and compute embeddings for all icons
 */
async function computeAllEmbeddings() {
    console.log("Loading model and data...");

    // Load the model
    const modelPath = "./models/icon-similarity-best/model.json";
    if (!fs.existsSync(modelPath)) {
        console.error("Model not found! Please train the model first.");
        process.exit(1);
    }

    const model = await tf.loadLayersModel(`file://${modelPath}`);
    console.log("Model loaded successfully");

    // Load processed data
    const {metadata, features} = loadProcessedData();
    const inputShape = getInputShape();

    console.log(`Computing embeddings for ${features.length} icons...`);

    // Compute embeddings in batches
    const batchSize = 32;
    const embeddings = [];

    for (let i = 0; i < features.length; i += batchSize) {
        const batch = features.slice(i, i + batchSize);
        const batchTensor = tf.tidy(() => {
            const tensors = batch.map(feat => tf.tensor3d(feat, inputShape));
            return tf.stack(tensors);
        });

        const batchEmbeddings = model.predict(batchTensor);
        // Normalize embeddings since the model doesn't have built-in normalization
        const normalizedEmbeddings = normalizeEmbeddings(batchEmbeddings);
        const embeddingData = await normalizedEmbeddings.data();

        // Store embeddings
        for (let j = 0; j < batch.length; j++) {
            const start = j * 128; // embedding dimension
            const end = start + 128;
            embeddings.push(Array.from(embeddingData.slice(start, end)));
        }

        batchTensor.dispose();
        batchEmbeddings.dispose();
        normalizedEmbeddings.dispose();

        if ((i + batchSize) % 320 === 0) {
            console.log(`Processed ${Math.min(i + batchSize, features.length)}/${features.length} icons`);
        }
    }

    return {embeddings, metadata};
}

/**
 * Find most similar icons to a given index
 */
function findSimilarIcons(queryIdx, embeddings, topK = 10) {
    const queryEmbedding = embeddings[queryIdx];
    const similarities = [];

    // Calculate cosine similarity with all other icons
    for (let i = 0; i < embeddings.length; i++) {
        if (i === queryIdx) continue;

        let dotProduct = 0;
        let normQuery = 0;
        let normOther = 0;

        for (let j = 0; j < queryEmbedding.length; j++) {
            dotProduct += queryEmbedding[j] * embeddings[i][j];
            normQuery += queryEmbedding[j] * queryEmbedding[j];
            normOther += embeddings[i][j] * embeddings[i][j];
        }

        normQuery = Math.sqrt(normQuery);
        normOther = Math.sqrt(normOther);

        const similarity = normQuery > 0 && normOther > 0 ? dotProduct / (normQuery * normOther) : 0;

        similarities.push({
            index: i,
            similarity: similarity
        });
    }

    // Sort by similarity and return top K
    similarities.sort((a, b) => b.similarity - a.similarity);
    return similarities.slice(0, topK);
}

/**
 * Test the similarity model
 */
async function testSimilarity() {
    const {embeddings, metadata} = await computeAllEmbeddings();

    console.log("\n=== Testing Icon Similarity ===");
    console.log("Testing with random icons...\n");

    // Test with several random icons
    const numTests = 5;
    for (let test = 0; test < numTests; test++) {
        const queryIdx = Math.floor(Math.random() * embeddings.length);
        const queryIcon = metadata.icons[queryIdx];

        console.log(`\nQuery Icon: ${queryIcon.name} (index: ${queryIdx})`);
        console.log("Most similar icons:");

        const similar = findSimilarIcons(queryIdx, embeddings, 10);

        similar.forEach((result, rank) => {
            const icon = metadata.icons[result.index];
            console.log(`  ${rank + 1}. ${icon.name} (similarity: ${result.similarity.toFixed(4)})`);
        });
    }

    // Calculate some statistics
    console.log("\n=== Embedding Statistics ===");

    // Average similarity between random pairs
    let totalSim = 0;
    const numPairs = 1000;
    for (let i = 0; i < numPairs; i++) {
        const idx1 = Math.floor(Math.random() * embeddings.length);
        const idx2 = Math.floor(Math.random() * embeddings.length);
        if (idx1 === idx2) continue;

        let dotProduct = 0;
        let norm1 = 0;
        let norm2 = 0;

        for (let j = 0; j < embeddings[0].length; j++) {
            dotProduct += embeddings[idx1][j] * embeddings[idx2][j];
            norm1 += embeddings[idx1][j] * embeddings[idx1][j];
            norm2 += embeddings[idx2][j] * embeddings[idx2][j];
        }

        const similarity = dotProduct / (Math.sqrt(norm1) * Math.sqrt(norm2));
        totalSim += similarity;
    }

    console.log(`Average random pair similarity: ${(totalSim / numPairs).toFixed(4)}`);

    // Check embedding norms
    let totalNorm = 0;
    embeddings.forEach(emb => {
        let norm = 0;
        emb.forEach(val => norm += val * val);
        totalNorm += Math.sqrt(norm);
    });
    console.log(`Average embedding norm: ${(totalNorm / embeddings.length).toFixed(4)}`);

    // Save embeddings for later use
    const embeddingData = {
        embeddings,
        metadata: {
            totalIcons: metadata.totalIcons,
            embeddingDim: embeddings[0].length,
            computedAt: new Date().toISOString()
        }
    };

    fs.writeFileSync("./icon-embeddings.json", JSON.stringify(embeddingData));
    console.log("\nEmbeddings saved to ./icon-embeddings.json");
}

// Run the test
testSimilarity().catch(console.error);