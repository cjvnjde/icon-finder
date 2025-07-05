import tf from "@tensorflow/tfjs-node";
import {augmentImageTensor} from "./augmentImageTensor.js";
import {getInputShape} from "./preprocess.js";

export function generateTriplets(features, numTriplets, embeddingDim) {
    return tf.tidy(() => {
        const inputShape = getInputShape();

        console.log(`Generating ${numTriplets} triplets...`);

        // Pre-allocate arrays for better performance
        const anchorData = [];
        const positiveData = [];
        const negativeData = [];

        for (let i = 0; i < numTriplets; i++) {
            // Random selection of anchor, positive, and negative
            const anchorIdx = Math.floor(Math.random() * features.length);

            let positiveIdx = Math.floor(Math.random() * features.length);
            while (positiveIdx === anchorIdx) {
                positiveIdx = Math.floor(Math.random() * features.length);
            }

            let negativeIdx = Math.floor(Math.random() * features.length);
            while (negativeIdx === anchorIdx || negativeIdx === positiveIdx) {
                negativeIdx = Math.floor(Math.random() * features.length);
            }

            // Store the data (not tensors yet)
            anchorData.push(features[anchorIdx]);
            positiveData.push(features[positiveIdx]);
            negativeData.push(features[negativeIdx]);
        }

        // Now create all tensors at once and apply augmentation
        const anchorTensors = anchorData.map((data) => {
            const tensor = tf.tensor3d(data, inputShape);
            return augmentImageTensor(tensor);
        });

        const positiveTensors = positiveData.map((data) => {
            const tensor = tf.tensor3d(data, inputShape);
            return augmentImageTensor(tensor);
        });

        const negativeTensors = negativeData.map((data) => {
            const tensor = tf.tensor3d(data, inputShape);
            return augmentImageTensor(tensor);
        });

        // Stack all tensors
        const anchorStack = tf.stack(anchorTensors);
        const positiveStack = tf.stack(positiveTensors);
        const negativeStack = tf.stack(negativeTensors);

        // Concatenate into single tensor [anchors, positives, negatives]
        const x = tf.concat([anchorStack, positiveStack, negativeStack], 0);

        // Create dummy labels (not used but required by Keras)
        const y = tf.zeros([numTriplets * 3, embeddingDim]);

        return { x, y };
    });
}
