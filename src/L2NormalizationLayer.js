import tf from "@tensorflow/tfjs-node";

/**
 * A custom layer to perform L2 normalization on the input tensor.
 * This is the modern equivalent for recent TensorFlow.js versions.
 */
export class L2NormalizationLayer extends tf.layers.Layer {
    constructor(config) {
        super(config || {});
        // The axis along which to normalize. For a batch of feature vectors, this is the feature axis.
        this.axis = -1;
    }

    call(input) {
        return tf.tidy(() => {
            const [inputTensor] = input;

            // --- FINAL CORRECTION: Manually compute L2 normalization ---
            // Calculate the L2 norm along the specified axis.
            // The 'true' argument keeps the number of dimensions the same for broadcasting.
            const norm = tf.norm(inputTensor, 'euclidean', this.axis, true);

            // Add a small epsilon value to prevent division by zero.
            const epsilon = tf.scalar(1e-8);

            // Divide the input tensor by its norm.
            return inputTensor.div(norm.add(epsilon));
        });
    }

    // This method helps TensorFlow.js infer the output shape.
    computeOutputShape(inputShape) {
        return inputShape;
    }

    // This is required for saving/loading the layer configuration.
    getConfig() {
        const config = super.getConfig();
        Object.assign(config, {axis: this.axis});
        return config;
    }

    static get className() {
        return 'L2NormalizationLayer';
    }
}

// Registration remains the same.
tf.serialization.registerClass(L2NormalizationLayer);
