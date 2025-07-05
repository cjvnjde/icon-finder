import tf from "@tensorflow/tfjs-node";

/**
 * Create a stable triplet loss function
 * @param {number} margin - Margin for triplet loss
 * @returns {Function} Loss function
 */
export function createTripletLoss(margin = 0.5) {
    return (_, yPred) => {
        return tf.tidy(() => {
            const batchSize = yPred.shape[0];
            const embeddingDim = yPred.shape[1];
            const numTriplets = Math.floor(batchSize / 3);

            if (numTriplets === 0) {
                return tf.scalar(0);
            }

            const anchors = yPred.slice([0, 0], [numTriplets, embeddingDim]);
            const positives = yPred.slice(
                [numTriplets, 0],
                [numTriplets, embeddingDim],
            );
            const negatives = yPred.slice(
                [numTriplets * 2, 0],
                [numTriplets, embeddingDim],
            );

            // L2 normalize embeddings
            const epsilon = 1e-8;
            const anchorNorm = tf.norm(anchors, 2, 1, true).add(epsilon);
            const positiveNorm = tf.norm(positives, 2, 1, true).add(epsilon);
            const negativeNorm = tf.norm(negatives, 2, 1, true).add(epsilon);

            const normalizedAnchors = anchors.div(anchorNorm);
            const normalizedPositives = positives.div(positiveNorm);
            const normalizedNegatives = negatives.div(negativeNorm);

            const posDistance = tf.norm(
                normalizedAnchors.sub(normalizedPositives),
                2,
                1,
            );
            const negDistance = tf.norm(
                normalizedAnchors.sub(normalizedNegatives),
                2,
                1,
            );

            const losses = tf.maximum(
                tf.scalar(0),
                posDistance.sub(negDistance).add(margin),
            );

            return tf.mean(losses);
        });
    };
}
