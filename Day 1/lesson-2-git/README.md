# Hands-on: Git for Version Control in Machine Learning Projects

# Test with class

## Objectives
By the end of this lab, you will:
- Install and configure Git
- Set up and manage a GitHub repository
- Track code changes with Git
- Use Git branching and merging
- Resolve merge conflicts
- Collaborate through GitHub
- Use the Fork GUI tool to manage version control

## Prerequisites
- A GitHub account
- Git installed on your machine
- VSCode or any code editor of your choice

## Files
- `app.py`: Script to train and evaluate the model.
- `iris_model.pkl`: Saved model after training.
- `requirements.txt`: Python dependencies.

## Steps to follow

### Step 1: Install and Verify Git
1. Download and install Git from git-scm.com.
2. Open your terminal and run: `git --version`

**Expected Outcome:** You should see the installed version of Git printed.

### Step 2: 
**Option 1: Create a GitHub Repository and Add scripts to it**
1. Go to GitHub and create a new repository (e.g., ml-git-lab)
2. Open terminal:
   ```bash
   git clone https://github.com/your-username/ml-git-lab.git
   ```
3. Open the folder and add scripts to it
   ```bash
   cd ml-git-lab
   ```

**Expected Outcome:** Your local folder contains the Git repository.

**Option 2: Clone the folder directly from github**
```bash
git clone https://github.com/AmmarMohanna/oreilly-mlops-bootcamp.git
cd oreilly-mlops-bootcamp/Day1/lesson-2-git
```

### Step 3: Modify Code and Commit
**Scenario:** You want to change test_size=0.2 to test_size=0.3

1. Check the current status of your Git repository
   ```bash
   git status
   ```
   - **Expected Output:** "nothing to commit, working tree clean" This means no changes have been made yet.

2. Make a code change
   - Open the file app.py
   - Locate the line where the dataset is split (e.g., test_size=0.2)
   - Change the test size to 0.3
   - Save the file
   - Check the Git status again: You should see app.py listed under "modified".

3. Stage the changes
   - **Option A (stage all changes):**
     ```bash
     git add .
     ```
   - **Option B (stage only app.py):**
     ```bash
     git add app.py
     ```

4. Commit the change with a meaningful message
   ```bash
   git commit -m "Changed test split size to 0.3"
   ```

### Step 4: Push to GitHub
```bash
git push origin main
```

**Expected Outcome:** Your changes are now reflected in your GitHub repository.

### Step 5: Branching and Feature Development
1. Create and switch to a new branch:
   ```bash
   git checkout -b logistic-model
   ```
2. Import the Logistic Regression model, train it, and save the new model
3. Stage and commit changes
4. Push to github `git push origin logistic-model`
5. Go to GitHub and open a Pull Request

## How to Use
1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
2. Run the model script:
   ```bash
   python app.py
   ```
3. Use `git init`, `git add`, and `git commit` to version your changes.

## Final Summary
In this hands-on, you successfully practiced the essential Git workflow: checking the repository status, staging modified files, committing changes with a clear message, and pushing updates to a remote GitHub repository. These are fundamental steps in version control that ensure your work is safely tracked, documented, and synchronized with collaborators. Mastering this workflow is crucial for maintaining clean, collaborative, and professional software projects.