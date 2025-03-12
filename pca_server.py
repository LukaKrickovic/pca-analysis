from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
import os
import uuid
from io import StringIO

app = FastAPI(title="PCA Analysis API")

STATIC_DIR = "static"
os.makedirs(STATIC_DIR, exist_ok=True)

# Mount the static directory to serve files
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

@app.get("/")
async def read_root():
    return FileResponse(f"{STATIC_DIR}/index.html")

@app.post("/analyze-pca/")
async def analyze_pca(file: UploadFile = File(...)):
    try:
        contents = await file.read()
        content_str = contents.decode('utf-8')
        
        df = pd.read_csv(StringIO(content_str))
        
        row_names = df.iloc[:, 0]
        df_data = df.iloc[:, 1:]
        df_data.index = row_names
        
        scaler = StandardScaler()
        scaled_data = scaler.fit_transform(df_data)
        
        pca = PCA(random_state=42)
        pca_result = pca.fit_transform(scaled_data)
        
        analysis_id = str(uuid.uuid4())[:8]
        
        plt.figure(figsize=(6, 5), dpi=300)  # Slightly larger figure for readability
        
        # Create scatter plot
        plt.scatter(pca_result[:, 0], pca_result[:, 1], color='black')
        
        # Annotate each point with its row name
        for i, txt in enumerate(row_names):
            plt.annotate(txt, 
                        (pca_result[i, 0], pca_result[i, 1]),
                        fontsize=8,
                        ha='right',  # horizontal alignment
                        textcoords='offset points',
                        xytext=(-2, 5),  # text offset in points
                        bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.7))
        
        # Add loadings vectors
        loadings = pca.components_.T * np.sqrt(pca.explained_variance_)
        for i, (x, y) in enumerate(zip(loadings[:, 0], loadings[:, 1])):
            plt.arrow(0, 0, x*0.95, y*0.95, color='red', alpha=0.5,
                     head_width=0.05, head_length=0.05)
            plt.text(x, y, df_data.columns[i], fontsize=9, color='red')
        
        plt.grid()
        plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.2%})')
        plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.2%})')
        plt.title('PCA Biplot')
        
        # Add tight layout to maximize the plot space
        plt.tight_layout()
        
        pca_plot_path = f"{STATIC_DIR}/PCA_{analysis_id}.png"
        plt.savefig(pca_plot_path)
        plt.close()
        
        # Create correlation plot (similar to corrplot.mixed in R)
        plt.figure(figsize=(5, 4), dpi=300)
        corr_matrix = df_data.corr()
        sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', vmin=-1, vmax=1)
        plt.title('Correlation Matrix')
        
        # Save correlation plot
        corr_plot_path = f"{STATIC_DIR}/Correlation_{analysis_id}.png"
        plt.savefig(corr_plot_path)
        plt.close()
        
        # Scree plot (similar to fviz_eig in R)
        plt.figure(figsize=(5, 4), dpi=300)
        explained_variance = pca.explained_variance_ratio_ * 100
        plt.bar(range(1, len(explained_variance) + 1), explained_variance)
        plt.plot(range(1, len(explained_variance) + 1), np.cumsum(explained_variance), 'ro-')
        plt.xlabel('Principal Components')
        plt.ylabel('Explained Variance (%)')
        plt.title('Scree Plot')
        plt.grid()
        
        # Save scree plot
        scree_plot_path = f"{STATIC_DIR}/Scree_{analysis_id}.png"
        plt.savefig(scree_plot_path)
        plt.close()
        
        # Return links to the plots
        return JSONResponse(content={
            "success": True,
            "pca_plot": f"/static/PCA_{analysis_id}.png",
            "correlation_plot": f"/static/Correlation_{analysis_id}.png",
            "scree_plot": f"/static/Scree_{analysis_id}.png",
            "explained_variance": {
                f"PC{i+1}": f"{var:.2f}%" 
                for i, var in enumerate(explained_variance)
            }
        })
    
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"success": False, "error": str(e)}
        )
