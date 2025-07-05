import tf from "@tensorflow/tfjs-node";

export function createTripletLoss(margin = 0.2) {
    return (_, yPred) => {
        return tf.tidy(() => {
            const batchSize = yPred.shape[0];
            const embeddingDim = yPred.shape[1];
            const numTriplets = Math.floor(batchSize / 3);

            if (numTriplets === 0) {
                return tf.scalar(0);
            }

            // Split the batch into anchors, positives, and negatives
            const anchors = yPred.slice([0, 0], [numTriplets, embeddingDim]);
            const positives = yPred.slice(
                [numTriplets, 0],
                [numTriplets, embeddingDim],
            );
            const negatives = yPred.slice(
                [numTriplets * 2, 0],
                [numTriplets, embeddingDim],
            );

            // L2 normalize embeddings here since we can't use lambda layer
            const epsilon = 1e-8;
            const anchorNorm = tf.norm(anchors, 2, 1, true).add(epsilon);
            const positiveNorm = tf.norm(positives, 2, 1, true).add(epsilon);
            const negativeNorm = tf.norm(negatives, 2, 1, true).add(epsilon);

            const normalizedAnchors = anchors.div(anchorNorm);
            const normalizedPositives = positives.div(positiveNorm);
            const normalizedNegatives = negatives.div(negativeNorm);

            // Calculate distances on normalized embeddings
            const posDistance = tf.sqrt(
                tf.sum(tf.square(tf.sub(normalizedAnchors, normalizedPositives)), 1).add(1e-8)
            );
            const negDistance = tf.sqrt(
                tf.sum(tf.square(tf.sub(normalizedAnchors, normalizedNegatives)), 1).add(1e-8)
            );

            // Calculate triplet loss with soft margin
            // Using soft margin can help with gradient flow
            const losses = tf.softplus(tf.add(tf.sub(posDistance, negDistance), margin));

            // Calculate percentage of non-zero losses for monitoring
            const nonZeroLosses = tf.greater(losses, 0.01);
            const percentActive = tf.mean(tf.cast(nonZeroLosses, 'float32'));

            // Log statistics occasionally
            if (Math.random() < 0.01) { // Log 1% of the time
                const stats = tf.tidy(() => {
                    return {
                        avgPosDistance: tf.mean(posDistance).dataSync()[0],
                        avgNegDistance: tf.mean(negDistance).dataSync()[0],
                        percentActive: percentActive.dataSync()[0] * 100,
                    };
                });
                console.log(`  Distance stats: pos=${stats.avgPosDistance.toFixed(3)}, neg=${stats.avgNegDistance.toFixed(3)}, active=${stats.percentActive.toFixed(1)}%`);
            }

            return tf.mean(losses);
        });
    };
}
