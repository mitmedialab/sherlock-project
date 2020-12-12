# HOW-TO: Mirror from a public to a private repository

This is a mirror repository based on [this](https://github.com/mitmedialab/sherlock-project) repository.

The creation of this mirror was performed using these [original instructions](https://medium.com/@bilalbayasut/github-how-to-make-a-fork-of-public-repository-private-6ee8cacaf9d3)

Ongoing top-ups can be made using the following:

```
# To pull new hotness from the public repo:

cd private-repo   # i.e. this folder
git remote add public https://github.com/mitmedialab/sherlock-project.git
git pull public master # Creates a merge commit
git push origin master
```
