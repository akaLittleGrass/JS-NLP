function encode(tokens) {
  let encodingMap = {};
  let decodingMap = {};
  let index = 0;
  tokens.map(token => {
    if (!encodingMap.hasOwnProperty(token)) {
      const pair = {};
      const unpair = {};
      pair[token] = index;
      unpair[index] = token;
      encodingMap = { ...encodingMap, ...pair };
      decodingMap = { ...decodingMap, ...unpair };
      index++;
    }
  });
  return {
    map: encodingMap,
    count: index,
    encode: function (word) {
      return encodingMap[word];
    },
    decode: function (index) {
      return decodingMap[index];
    }
  };
};

function generateData(words) {
  const tokens = words.match(/[^\s\.]+/g);
  const encoding = encode(tokens);
  const vocabSize = encoding.count;
  const data = [];
  const windowSize = 2 + 1;
  for (let i = 0; i < tokens.length; i++) {
    const token = tokens[i];
    for (let j = i - windowSize; j < i + windowSize; j++) {
      if (j >= 0 && j != i && j < tokens.length) {
        data.push([token, tokens[j]]);
      }
    }
  };
  const xTrainData = [];
  const yTrainData = [];
  data.map(pair => {
    const x = tf.oneHot(encoding.encode(pair[0]), vocabSize);
    const y = tf.oneHot(encoding.encode(pair[1]), vocabSize);
    xTrainData.push(x);
    yTrainData.push(y);
  });
  const xTrain = tf.stack(xTrainData);
  const yTrain = tf.stack(yTrainData);
  return {
    xTrain,
    yTrain,
    vocabSize,
    tokens
  };
};
export { generateData , encode};