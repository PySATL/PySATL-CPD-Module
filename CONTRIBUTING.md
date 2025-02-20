# PySATL CPD project contributing guide

Thank you very much if you have decided to contribute to our project.
We follow very simple and clear open-source research community accepted guidelines for contributing.
The guideline instructions divided into sections depending on the part of the project you want to contribute.

## Rules for adding commits

Create a new branch, if you want to add something new.
Recommended naming branch is `<type>/<name of stuff>`.

Commits are added according to conventional commits.
Those `<type>(<scope>): <body>`.

The `<type>` field must take one of these values:

* `feat` to add new functionality
* `fix` to fix a bug in the project
* `refactor` for code refactoring, such as renaming a variable
* `test` to add tests, refactor them
* `struct` for changes related to a change in the structure of the project (BUT NOT CODE), for example, changing
  folder locations
* `ci` for various ci/cd tasks
* `docs` for changes in documentation

The `<body>` field contains the gist of the changes in the present imperative in English without the dot in at the end,
the first word is a verb with a small letter.

Examples:

* Good: "feat: add module for future scrubber implementations"
* Bad: "Added module for future scrubber implementations."

## Source code developers guide

1. Fork this repository using your GitHub account.
2. Install `git` and clone your forked copy of the `repo`.
3. Build project following build instructions in [README.md](./README.md) file, make sure everything is ok.
4. Run tests following instructions in [README.md](./README.md) file, make sure all tests passing.
5. Implement new feature or fix existing one in the source code.
6. Commit your changes.
7. Open a pull-request.
8. Wait for review from developers of the project.
9. Fix major and minor issues if presented.
10. Get your work merged into `main`!

## Rules for collaborators

### Basic Tips

1. Don't use merge, only rebase (to keep a linear commit history)
2. Do not change other people's branches unless absolutely necessary
3. Recheck your commit history before creating a pull request
4. **Check you're on the right branch**, never commit directly in main

### Rules for pull requests

**Forbidden** to merge your pull request into the branch yourself.

Each pull request must be reviewed by one of the maintainers

* Alexey Tatyanenko ([alexdtat](https://github.com/alexdtat))
* Artemii Patov ([artemiipatov](https://github.com/artemiipatov))
* Vladimir Kutuev ([vkutuev](https://github.com/vkutuev))

If you click on the green button, then **make sure** that it says `REBASE AND MERGE`!

The review takes place in the form of comments to pull requests, discussions in the team chat and personal
communication.
