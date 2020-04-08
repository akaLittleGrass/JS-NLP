import Model from './Model.js';
import Data from './DataSet.js';

const BATCHSIZE = 16;
const EPOCHS = 500;

const model = Model();
model.fit(xTrain, yTrain, {
  BATCHSIZE,
  EPOCHS,//迭代次数
  shuffle: true,//是否乱序
});