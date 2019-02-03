import * as tf from "@tensorflow/tfjs";

export const createNN = ({ inputs, ouputs, layers, layerNodes }) => {
  const model = tf.sequential();
  //Input Layer
  model.add(
    tf.layers.dense({ units: layerNodes, useBias: true, inputDim: inputs, kernelInitializer: "truncatedNormal", biasInitializer: "truncatedNormal" })
  );

  //Hidden Layers
  for (let i = 0; i < layers - 2; i++) {
    model.add(tf.layers.dense({ units: layerNodes, useBias: true, kernelInitializer: "truncatedNormal", biasInitializer: "truncatedNormal" }));
  }

  //Output Layers
  model.add(
    tf.layers.dense({ units: ouputs, useBias: true, activation: "sigmoid", kernelInitializer: "truncatedNormal", biasInitializer: "truncatedNormal" })
  );

  return model;
};

const mutateWeights = (weights, mutationRate) => {
  const uniform = tf.randomUniform(weights.shape);
  // console.log('uniform')
  // uniform.print()
  const mutationRateMat = tf.fill(weights.shape, mutationRate);
  // console.log('mutationRateMat')
  // mutationRateMat.print()
  const mutatedMask = uniform.lessEqual(mutationRateMat);
  // mutatedMask.print();
  const notMutatedMask = tf.logicalNot(mutatedMask);

  const mutationsMat = tf.truncatedNormal(weights.shape, 0, 0.05);

  const first = mutationsMat.mul(mutatedMask);
  const second = weights.mul(notMutatedMask);
  return first.add(second);
};

export const mutateNN = async ({ nn, mutationRate = 0.05 }) => {
  const mutatedNN = await copyNN(nn);
  const layersNum = mutatedNN.layers.length;
  for (let i = 0; i < layersNum; i++) {
    const cLayer = mutatedNN.getLayer(null, i);
    const weights = cLayer.getWeights();

    const kernel = mutateWeights(weights[0], mutationRate);
    const bias = mutateWeights(weights[1], mutationRate);
    cLayer.setWeights([kernel, bias]);
  }
  return mutatedNN;
};

export const copyNN = async nn => {
  const tempName = "nn_" + Date.now();
  await nn.save("localstorage://" + tempName);
  const loadedModel = await tf.loadModel("localstorage://" + tempName);
  await tf.io.removeModel("localstorage://" + tempName);
  return loadedModel;
};
