document.addEventListener("DOMContentLoaded", function () {
  const uploadBtn = document.getElementById("uploadBtn");
  const csvFileInput = document.getElementById("csvFile");
  const loadingElement = document.getElementById("loading");
  const resultsElement = document.getElementById("results");
  const pcaPlotElement = document.getElementById("pcaPlot");
  const corrPlotElement = document.getElementById("corrPlot");
  const screePlotElement = document.getElementById("screePlot");
  const explainedVarianceElement = document.getElementById("explainedVariance");

  uploadBtn.addEventListener("click", async function () {
    const file = csvFileInput.files[0];
    if (!file) {
      alert("Please select a CSV file first");
      return;
    }

    // Display loading spinner
    loadingElement.classList.remove("hidden");
    resultsElement.classList.add("hidden");

    try {
      const formData = new FormData();
      formData.append("file", file);

      const response = await fetch("/analyze-pca/", {
        method: "POST",
        body: formData,
      });

      const data = await response.json();

      if (data.success) {
        // Update the plots
        pcaPlotElement.src = data.pca_plot;
        corrPlotElement.src = data.correlation_plot;
        screePlotElement.src = data.scree_plot;

        // Display explained variance
        explainedVarianceElement.innerHTML = "";
        Object.entries(data.explained_variance).forEach(([pc, variance]) => {
          const varianceItem = document.createElement("div");
          varianceItem.className = "variance-item";
          varianceItem.textContent = `${pc}: ${variance}`;
          explainedVarianceElement.appendChild(varianceItem);
        });

        // Show results
        resultsElement.classList.remove("hidden");
      } else {
        alert(`Error: ${data.error}`);
      }
    } catch (error) {
      alert(`An error occurred: ${error.message}`);
    } finally {
      // Hide loading spinner
      loadingElement.classList.add("hidden");
    }
  });
});
