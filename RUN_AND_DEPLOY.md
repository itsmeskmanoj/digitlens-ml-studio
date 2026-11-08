# Run, GitHub and deployment steps

## 1. Run on BITS Virtual Lab

Open a terminal in the project folder and run:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python train_models.py
streamlit run app.py
```

If the lab is Windows, activate using:

```powershell
.venv\Scripts\activate
```

Open the Streamlit URL, choose a model, use `test_data.csv`, and capture **one screenshot** showing that the assignment executed in BITS Virtual Lab.

## 2. Push updates to GitHub

The repository is already configured at https://github.com/itsmeskmanoj/digitlens-ml-studio.
Review the changes before committing, then run:

```bash
git status
git add app.py train_models.py requirements.txt model test_data.csv model_metrics.csv model_metadata.json README.md RUN_AND_DEPLOY.md .gitignore
git diff --cached --stat
git commit -m "Add digit classification app, trained models, and deployment guide"
git push origin main
```

Use meaningful commit messages for subsequent changes because the assignment states that commit history may be reviewed. Never commit your virtual environment or credentials.

## 3. Update README links

The GitHub link is already set. After deployment, replace the "Not deployed yet" status in `README.md` with the live Streamlit URL, then commit and push:

```bash
git add README.md
git commit -m "Add final repository and deployment links"
git push
```

## 4. Deploy on Streamlit Community Cloud

1. Visit https://streamlit.io/cloud and sign in with GitHub.
2. Select **New app**.
3. Select your repository and the `main` branch.
4. Set the entrypoint to `app.py`.
5. Click **Deploy**.
6. Test the live app using `test_data.csv`.

## 5. Final submission PDF order

The PDF must contain, in this exact order:

1. Clickable GitHub repository link
2. Clickable live Streamlit application link
3. BITS Virtual Lab execution screenshot
4. Complete README content, including the metrics table and observations

Do not submit as a draft. Use the final **Submit assignment** action in Taxila.
