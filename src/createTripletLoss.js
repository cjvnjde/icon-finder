// file: src/createTripletLoss.js
import tf from "@tensorflow/tfjs-node";

export function createTripletLoss(margin = 0.2) {
    return (_, yPred) => {
        return tf.tidy(() => {
            const numTriplets = Math.floor(yPred.shape[0] / 3);

            // Because the model now outputs L2-normalized embeddings, we don't need to normalize here.
            const anchors = yPred.slice([0, 0], [numTriplets, -1]);
            const positives = yPred.slice([numTriplets, 0], [numTriplets, -1]);
            const negatives = yPred.slice([numTriplets * 2, 0], [numTriplets, -1]);

            // Calculate squared Euclidean distance. It's computationally cheaper and works just as well.
            const posDistance = tf.sum(tf.square(tf.sub(anchors, positives)), 1);
            const negDistance = tf.sum(tf.square(tf.sub(anchors, negatives)), 1);

            // --- IMPROVEMENT: Use standard hinge loss (max(0, ...)) ---
            const losses = tf.maximum(0, tf.add(tf.sub(posDistance, negDistance), margin));

            return tf.mean(losses);
        });
    };
}
