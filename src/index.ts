import "bootstrap/dist/css/bootstrap.min.css";
import * as maskedlm from "../maskedlm/pkg";

const button = document.getElementById("button");

button.addEventListener('click', function(event) {
  const input = document.getElementById("input").value;
  const output = maskedlm.predict_masked_words(input);
  document.getElementById("output").value = output;
});
