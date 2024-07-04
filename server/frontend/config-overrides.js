const { override, addBabelPreset } = require('customize-cra');

module.exports = override(
  addBabelPreset('@babel/preset-react')
  // 여기에 추가적인 설정을 넣을 수 있습니다.
  // 예: addWebpackPlugin(), addWebpackAlias() 등
);