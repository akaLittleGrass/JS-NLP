export default function (inputSize, outputSize) {
    const model = tf.sequential({
        layers: [
            tf.layers.dense({
                units: 2,
                inputShape: inputSize,
                name: 'embedding'
            }),
            tf.layers.dense({
                units: outputSize,
                kernelInitializer: 'varianceScaling', activation: 'softmax'
            })
        ]
    });
    model.compile({
        optimizer: tf.train.adam(),
        loss: tf.losses.softmaxCrossEntropy,
        metrics: ['accuracy'],
    });
    return model;
}