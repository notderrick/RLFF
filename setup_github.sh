#!/bin/bash
# GitHub Setup Helper Script

echo "RLFF GitHub Setup"
echo "================="
echo ""

# Check if gh is authenticated
if gh auth status &> /dev/null; then
    echo "✓ GitHub CLI is authenticated"
    echo ""
    echo "Creating repository..."
    gh repo create RLFF --public --source=. \
        --description "Reinforcement Learning Fantasy Football - An RL-powered draft agent that learns strategic drafting through GRPO" \
        --push

    if [ $? -eq 0 ]; then
        echo ""
        echo "✓ Repository created and pushed successfully!"
        echo ""
        REPO_URL=$(gh repo view --json url -q .url)
        echo "Repository URL: $REPO_URL"
        echo ""
        echo "Next steps:"
        echo "  1. Visit: $REPO_URL"
        echo "  2. Add topics: machine-learning, reinforcement-learning, fantasy-football, nlp"
        echo "  3. Start testing: python test_env.py"
    fi
else
    echo "✗ GitHub CLI not authenticated"
    echo ""
    echo "Option 1: Authenticate with GitHub CLI (Recommended)"
    echo "  gh auth login"
    echo ""
    echo "Option 2: Manual Setup"
    echo "  1. Create repo at: https://github.com/new"
    echo "     Name: RLFF"
    echo "     Description: Reinforcement Learning Fantasy Football - An RL-powered draft agent"
    echo "     Public: Yes"
    echo ""
    echo "  2. Run these commands:"
    echo "     git remote add origin https://github.com/YOUR_USERNAME/RLFF.git"
    echo "     git push -u origin main"
    echo ""
fi
