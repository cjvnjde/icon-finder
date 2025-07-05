import tf from "@tensorflow/tfjs-node";

/**
 * Normalize embeddings to unit length (L2 normalization)
 * @param {tf.Tensor} embeddings - Tensor of embeddings (batch_size x embedding_dim)
 * @returns {tf.Tensor} Normalized embeddings
 */
export function normalizeEmbeddings(embeddings) {
    return tf.tidy(() => {
        const epsilon = 1e-8;
        const norm = tf.norm(embeddings, 2, 1, true).add(epsilon);
        return embeddings.div(norm);
    });
}

/**
 * Compute normalized embeddings from a model
 * @param {tf.LayersModel} model - The trained model
 * @param {tf.Tensor} input - Input tensor
 * @returns {tf.Tensor} Normalized embeddings
 */
export function computeNormalizedEmbeddings(model, input) {
    return tf.tidy(() => {
        const rawEmbeddings = model.predict(input);
        return normalizeEmbeddings(rawEmbeddings);
    });
}

/**
 * Calculate cosine similarity between two normalized embedding tensors
 * @param {tf.Tensor} embeddings1 - First embedding tensor
 * @param {tf.Tensor} embeddings2 - Second embedding tensor
 * @returns {tf.Tensor} Similarity scores
 */
export function cosineSimilarity(embeddings1, embeddings2) {
    return tf.tidy(() => {
        // Since embeddings are normalized, cosine similarity is just dot product
        return tf.sum(tf.mul(embeddings1, embeddings2), 1);
    });
}