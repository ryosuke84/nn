import "./styles.css";
import * as tf from "@tensorflow/tfjs";
import { createNN, mutateNN, copyNN } from "../src/nn";

document.getElementById("app").innerHTML = `
<h1>Hello Vanilla!</h1>
<div>
  We use Parcel to bundle this sandbox, you can find more info about Parcel
  <a href="https://parceljs.org" target="_blank" rel="noopener noreferrer">here</a>.
</div>
`;

const brain = createNN({ inputs: 5, ouputs: 2, layers: 2, layerNodes: 5 });
// console.log(brain);
const layer = brain.getLayer(null, 0);
const weights = layer.getWeights();
// weights[0].print();
// const before = weights[0].buffer();

mutateNN({ nn: brain, mutationRate: 0.05 }).then(mutatedBrain => {
  const mLayer = mutatedBrain.getLayer(null, 0);
  const mWeights = mLayer.getWeights();
  weights[0].print();
  mWeights[0].print();
  mWeights[0].notEqual(weights[0]).print();
});

// mWeights[0].notEqual(before.toTensor()).print();
// console.log(layer);
// console.log(weights);

// const buff = weights[1].buffer();
// for (let i = 0; i < buff.size; i++) {
//   buff.set(0, i);
// }

// weights[0] = buff.toTensor();
// weights[0].print();
// console.log(weights);

// ------------------------------------------------

// MUTATION
// const uniform = tf.randomUniform([10, 10]);
// // uniform.print();
// const mut = tf.fill([10, 10], 0.05);
// // uniform.print()
// const mask = uniform.lessEqual(mut);
// // mask.print();
// // uniform.mul(mask).print()
// const notMask = tf.logicalNot(mask);
// // notMask.print()

// const mutations = tf.truncatedNormal([10, 10], 0, 0.05);

// const first = mutations.mul(mask);
// const second = uniform.mul(notMask);
// const mutatedMat = first.add(second);

// ---------------------------------------------
// brain.summary();
// const p = brain.predict(tf.ones([1, 100]))
// p.print()

// const model = tf.sequential()
// const layer1 = tf.layers.dense({ units: 16, useBias: true, inputDim: 12000, kernelInitializer:'zeros' })
// const layer2 = tf.layers.dense({ units: 2, useBias: true, activation: 'sigmoid', kernelInitializer: 'zeros' })
// model.add(layer1) // input
// model.add(layer2) // hidden

// //layer1.getWeights()[0].print()
// const biases = layer1.getWeights()[1]

// const buffer = tf.buffer(biases.shape, biases.dtype, biases.dataSync());
// buffer.set(5, 0);
// const b = buffer.toTensor();
// // Convert the buffer back to a tensor.
// b.print()

// //Predicting
// for (let i = 0;i < 100; i++){
//   const p = model.predict(tf.ones([1, 12000]))
//   // p.print()
// }
// console.log('finished')

//---------- NN copy
// for(const layer of brain.layers) {
//   console.log(layer.getWeights())

// }
// const copyNN = async function(nn) {
//   console.log(Date.now());
//   const saveResult = await nn.save("localstorage://my-model-1");
//   const loadedModel = await tf.loadModel("localstorage://my-model-1");
//   await tf.io.removeModel("localstorage://my-model-1");

//   const layer = nn.getLayer(null, 0);
//   const weights = layer.getWeights();

//   const mutatedBrain = mutateNN({ nn: loadedModel, mutationRate: 1 });
//   const mLayer = mutatedBrain.getLayer(null, 0);
//   const mWeights = mLayer.getWeights();
//   weights[0].print();
//   mWeights[0].print();
// };

// copyNN(brain);

// const copyBrain = copyNN(brain).then(e => {
//   console.log(e);
// });
