function encode (tokens) {
  const encodingMap = {};
  const decodingMap = {};
  let index = 0;
  tokens.map( token => {
    if(!encodingMap.hasOwnProperty(token)) {
      const pair = {};
      const unpair = {};
      pair[token] = index;
      unpair[index] = token;
      encodingMap = {...encodingMap, ...pair}; 
      decodingMap = {...decodingMap, ...unpair}; 
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

export default function (words) {
    const tokens = words.match(/[^\s\.]+/g);
    const encoding = encode(tokens);
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
    const xTrain = [];
    const yTrain = [];
    data.map( pair => {
        x = tf.oneHot(encoding.encode(pair[0]), encoding.count);
        y = tf.oneHot(encoding.encode(pair[1]), encoding.count);
        xTrain.push(x);
        yTrain.push(y);
    });
    return {
        xTrain,
        yTrain
    };
}