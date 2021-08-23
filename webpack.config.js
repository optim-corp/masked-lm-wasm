const path = require("path");
const HtmlWebpackPlugin = require("html-webpack-plugin");
const WasmPackPlugin = require("@wasm-tool/wasm-pack-plugin");

module.exports = {
  resolve: {
    extensions: [".ts", ".tsx", ".js", ".jsx", ".wasm"]
  },
  mode: "production",
  module: {
    rules: [
      {
        test: /\.tsx?$/,
        loader: "ts-loader",
        options: {
          transpileOnly: true
        }
      },
      {
        test: /\.css$/,
        use: [
          "style-loader",
          "css-loader"
        ],
      }
    ]
  },
  plugins: [
    new HtmlWebpackPlugin({
      template: path.join(__dirname, "src/index.html")
    }),
    new WasmPackPlugin({
      crateDirectory: path.join(__dirname, "maskedlm")
    })
  ],
  experiments: {
    asyncWebAssembly: true,
  },
};
