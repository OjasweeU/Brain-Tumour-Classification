import { useState } from "react";

const API_BASE_URL = import.meta.env.VITE_API_BASE_URL ?? "http://localhost:8000";

function formatLabel(label) {
  return label.replaceAll("_", " ").replace(/\b\w/g, (char) => char.toUpperCase());
}

export default function App() {
  const [file, setFile] = useState(null);
  const [previewUrl, setPreviewUrl] = useState("");
  const [result, setResult] = useState(null);
  const [error, setError] = useState("");
  const [isLoading, setIsLoading] = useState(false);

  async function handleSubmit(event) {
    event.preventDefault();
    if (!file) {
      setError("Choose an MRI image before running analysis.");
      return;
    }

    const formData = new FormData();
    formData.append("file", file);

    setIsLoading(true);
    setError("");
    setResult(null);

    try {
      const response = await fetch(`${API_BASE_URL}/predict`, {
        method: "POST",
        body: formData
      });

      const payload = await response.json();
      if (!response.ok) {
        throw new Error(payload.detail || "Prediction failed.");
      }

      setResult(payload);
    } catch (requestError) {
      setError(requestError.message);
    } finally {
      setIsLoading(false);
    }
  }

  function handleFileChange(event) {
    const nextFile = event.target.files?.[0] ?? null;
    setFile(nextFile);
    setResult(null);
    setError("");

    if (nextFile) {
      setPreviewUrl(URL.createObjectURL(nextFile));
    } else {
      setPreviewUrl("");
    }
  }

  return (
    <main className="page-shell">
      <section className="hero">
        <div className="hero-copy">
          <p className="eyebrow">Full-Stack Medical Imaging Demo</p>
          <h1>Brain Tumor MRI Classifier</h1>
          <p className="lede">
            Upload a brain MRI scan and we&apos;ll run EfficientNet-based classification with a Grad-CAM
            explanation layer. The training workflow stays separate; this interface only handles inference.
          </p>
        </div>
        <div className="hero-panel">
          <div className="mini-stat">
            <span>Model classes</span>
            <strong>4</strong>
          </div>
          <div className="mini-stat">
            <span>Input size</span>
            <strong>240 x 240</strong>
          </div>
          <div className="mini-stat">
            <span>Explainability</span>
            <strong>Grad-CAM</strong>
          </div>
        </div>
      </section>

      <section className="workspace">
        <form className="upload-card" onSubmit={handleSubmit}>
          <div>
            <p className="section-tag">Inference</p>
            <h2>Upload MRI image</h2>
            <p className="muted">
              JPG, JPEG, or PNG. The backend applies the same crop-and-resize preprocessing used for the model.
            </p>
          </div>

          <label className="dropzone">
            <input type="file" accept="image/png,image/jpeg,image/jpg" onChange={handleFileChange} />
            {previewUrl ? (
              <img src={previewUrl} alt="MRI preview" className="dropzone-preview" />
            ) : (
              <div className="dropzone-copy">
                <strong>Select an MRI scan</strong>
                <span>Click to browse or drag an image here</span>
              </div>
            )}
          </label>

          <button type="submit" className="primary-button" disabled={isLoading}>
            {isLoading ? "Analyzing..." : "Analyze MRI"}
          </button>

          {error ? <p className="error-banner">{error}</p> : null}
        </form>

        <section className="results-card">
          <div className="results-header">
            <div>
              <p className="section-tag">Results</p>
              <h2>Prediction dashboard</h2>
            </div>
            {result ? (
              <div className="result-badge">
                <span>{formatLabel(result.predictedLabel)}</span>
                <strong>{(result.confidence * 100).toFixed(2)}%</strong>
              </div>
            ) : null}
          </div>

          {result ? (
            <>
              <div className="image-grid">
                <figure className="image-card">
                  <img src={result.images.original} alt="Original MRI" />
                  <figcaption>Uploaded MRI</figcaption>
                </figure>
                <figure className="image-card">
                  <img src={result.images.cropped} alt="Cropped MRI" />
                  <figcaption>Cropped brain region</figcaption>
                </figure>
                <figure className="image-card">
                  <img src={result.images.gradcam} alt="Grad-CAM heatmap" />
                  <figcaption>Grad-CAM overlay</figcaption>
                </figure>
              </div>

              <div className="probability-card">
                <h3>Class probabilities</h3>
                <div className="probability-list">
                  {result.probabilities.map((item) => (
                    <div className="probability-row" key={item["Tumor Type"]}>
                      <div className="probability-meta">
                        <span>{item["Tumor Type"]}</span>
                        <strong>{(item.Probability * 100).toFixed(2)}%</strong>
                      </div>
                      <div className="probability-track">
                        <div
                          className="probability-fill"
                          style={{ width: `${Math.max(item.Probability * 100, 4)}%` }}
                        />
                      </div>
                    </div>
                  ))}
                </div>
              </div>
            </>
          ) : (
            <div className="placeholder">
              <h3>Ready for inference</h3>
              <p>
                Once you upload an MRI image, we&apos;ll show the predicted tumor type, confidence scores, and Grad-CAM
                visual explanation here.
              </p>
            </div>
          )}
        </section>
      </section>
    </main>
  );
}
