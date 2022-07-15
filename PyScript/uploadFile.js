var inputElement = document.getElementById("input");
var reader = new FileReader();
var downloadLink = document.getElementById('downloadLink');

reader.onloadend = function(){
  console.log(reader.result);
}
inputElement.addEventListener("change", handleFiles, false);
function handleFiles() {
  var fileSelected = this.files[0]; /* now you can work with the file list */
  reader.readAsBinaryString(fileSelected);
  downloadLink.href = window.URL.createObjectURL(fileSelected);
}