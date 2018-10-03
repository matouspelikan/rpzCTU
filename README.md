# RPZ python assignment templates
## Using this repo
The recommended way of using the templates is as follows.

First, you clone the repository:
```bash
git clone git@gitlab.fel.cvut.cz:neoramic/rpz-python-assignment-templates.git
```

Then, you create a new branch for your solutions:
```bash
cd rpz-python-assignment-templates
git checkout -b solutions
```

After that, you can work on your solutions, commiting as necessary.

In order to update the template, commit all your work and execute:
```bash
# download the new template version:
git checkout master
git pull
# and update your solutions branch:
git checkout solutions
git merge master
```
