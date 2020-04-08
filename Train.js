import Model from './Model.js';
import { generateData, encode } from './DataSet.js';

const BATCHSIZE = 16;
const EPOCHS = 500;
const WORDS = 'May we hold tight the present for we may lose each other in the crowed once apart.';

const { xTrain, yTrain, vocabSize, tokens } = generateData(WORDS);

const model = Model(vocabSize, vocabSize);
const metrics = ['loss', 'val_loss', 'acc', 'val_acc'];
const container = {
  name: 'Model Training', styles: { height: '1000px' }
};
const fitCallbacks = tfvis.show.fitCallbacks(container, metrics);
tfvis.show.modelSummary({ name: 'Model Architecture' }, model);

model.fit(xTrain, yTrain, {
  BATCHSIZE,
  EPOCHS,//迭代次数
  shuffle: true,//是否乱序
  callbacks: fitCallbacks
});

const tokensData = [];
const encoding = encode(tokens);
tokens.map(token => {
  tokensData.push(tf.oneHot(encoding.encode(token), vocabSize));
});

const perdictInputs = tf.stack(tokensData);
//const perdictResult = model.predict(perdictInputs);
// tf.unstack(perdictResult).forEach(tensor => {
//   const index = tensor.argMax().dataSync()[0];
//   const result = encoding.decode(index);
// });

const embeddingLayer = model.getLayer('embedding');
const visModel = tf.sequential();
visModel.add(embeddingLayer);
const visResult = visModel.predict(perdictInputs).arraySync();

const vizData = [];
for (let i = 0; i < vocabSize; i++) {
  const word = encoding.decode(i);
  const pos = visResult[i];
  vizData.push({
    label: word,
    x: pos[0],
    y: pos[1]
  });
}
setTimeout(() => {
  const chart = new G2.Chart({
    container: 'chart',
    autoFit: false,
    width: 800,
    height: 800
  });
  chart.source(vizData);
  console.log(vizData);
  chart.point().position('x*y').label('label');
  chart.render();
})