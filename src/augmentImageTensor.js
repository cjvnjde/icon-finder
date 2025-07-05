import tf from "@tensorflow/tfjs-node";

/**
 * Simplified augmentation that avoids unsupported operations
 * @param {tf.Tensor} tensor - Image tensor (3D or 4D)
 * @returns {tf.Tensor} Augmented image tensor
 */
export function augmentImageTensor(tensor) {
    return tf.tidy(() => {
        // Ensure we have a 4D tensor
        let augmented = tensor.rank === 3 ? tf.expandDims(tensor, 0) : tensor.clone();

        // Random horizontal flip (50% chance)
        if (Math.random() > 0.5) {
            augmented = tf.reverse(augmented, 2); // axis 2 for width
        }

        // Random brightness adjustment (mild)
        if (Math.random() > 0.5) {
            const brightness = 0.9 + Math.random() * 0.2; // 0.9 to 1.1
            augmented = tf.clipByValue(tf.mul(augmented, brightness), 0, 1);
        }

        // Random contrast adjustment
        if (Math.random() > 0.5) {
            const contrast = 0.9 + Math.random() * 0.2; // 0.9 to 1.1
            const mean = tf.mean(augmented, [1, 2, 3], true);
            augmented = tf.clipByValue(
                tf.add(tf.mul(tf.sub(augmented, mean), contrast), mean),
                0, 1
            );
        }

        // Add subtle noise
        if (Math.random() > 0.6) {
            const noiseStrength = 0.02;
            const noise = tf.randomNormal(augmented.shape, 0, noiseStrength);
            augmented = tf.clipByValue(tf.add(augmented, noise), 0, 1);
        }

        // Simple zoom using resize (avoid complex transforms)
        if (Math.random() > 0.7) {
            const zoomFactor = 0.95 + Math.random() * 0.1; // 0.95 to 1.05
            const [batch, height, width, channels] = augmented.shape;

            if (zoomFactor < 1.0) {
                // Zoom out: resize smaller then pad
                const newSize = Math.floor(height * zoomFactor);
                const resized = tf.image.resizeBilinear(augmented, [newSize, newSize]);
                const padAmount = Math.floor((height - newSize) / 2);
                const paddings = [
                    [0, 0],
                    [padAmount, height - newSize - padAmount],
                    [padAmount, width - newSize - padAmount],
                    [0, 0]
                ];
                augmented = tf.pad(resized, paddings, 0);
            } else {
                // Zoom in: resize larger then center crop
                const newSize = Math.floor(height * zoomFactor);
                const resized = tf.image.resizeBilinear(augmented, [newSize, newSize]);
                const cropAmount = Math.floor((newSize - height) / 2);
                augmented = tf.slice(
                    resized,
                    [0, cropAmount, cropAmount, 0],
                    [batch, height, width, channels]
                );
            }
        }

        // Return with original rank
        return tensor.rank === 3 ? tf.squeeze(augmented, 0) : augmented;
    });
}
