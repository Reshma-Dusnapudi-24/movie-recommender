# 🎬 Movie Recommender System

An intelligent and interactive **content-based movie recommendation system** that suggests the top 5 most similar movies based on your selection. It uses NLP techniques and the OMDb API to enhance user experience with movie posters and details.

🔗 **Live Demo**: [Click here to try it out](https://netflix-movie-recommenders.streamlit.app/)  
📁 **Dataset**: Top 1000 IMDb Movies (`imdb_top_1000.csv`)

---

## 🚀 Features

- 📽️ Get 5 similar movies based on your selected movie
- 🖼️ Movie posters and descriptions fetched via **OMDb API**
- 📊 Uses **cosine similarity** for recommendations
- ⚡ Built with **Streamlit** for an interactive web experience
- 🌐 Hosted online using **Streamlit Cloud**

---

## 📂 Dataset Details

- Dataset: `imdb_top_1000.csv`
- Attributes used:
  - `Series_Title`, `Genre`, `Director`, `Star1`, `Star2`, `Star3`, `Star4`, `Overview`
- Focused on creating a combined feature set for similarity analysis.

---

## 🧰 Tech Stack

| Component        | Technology           |
|------------------|----------------------|
| Programming      | Python               |
| Web App          | Streamlit            |
| ML / NLP         | Pandas, Scikit-learn |
| API Integration  | OMDb API             |
| Hosting          | Streamlit Cloud      |

---

## 📌 How It Works

1. Loads IMDb Top 1000 movies dataset.
2. Preprocesses data and combines features (`overview`, `genre`, `cast`, etc.)
3. Vectorizes the combined text using **CountVectorizer**.
4. Calculates **cosine similarity** between all movies.
5. On user input, shows top 5 similar movies with posters and metadata.

---

## ⚙️ Installation & Setup

```bash
# Clone the repository
git clone https://github.com/YourUsername/Movie-Recommender.git
cd Movie-Recommender

# Install dependencies
pip install -r requirements.txt

# Run the application
streamlit run app.py
