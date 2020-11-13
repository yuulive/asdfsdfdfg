---
title: Continuous Integration
license: CC BY-SA 4.0
---

# Continuous integration

## Description

The use of GitLab platform for the management of the source code of the library, allows the management of the continuous integration activities.

The process of continuous integration allows to have the certainty that, after changes to the library, the code can compile and that all tests are executed.

The pipeline is executed every time that the changes to the source code are pushed to the repository. Failures during the process are reported by email.

## Structure

The pipeline is made by the following list of stage and jobs:

1. build
  - *build:oldest* library compilation with the minimum supported `Rust` version, both in debug and release mode
  - *build:latest* library compilation with the latest `Rust` version, both in debug and in release mode, it is executed also in merge request events

2. test
  - *test:run* execution of all tests present in the library with the latest `Rust` version, both in debug and in release mode, it is executed also in merge request events
  - *cover* execution of test coverage program `tarpaulin`

3. examples
  - *examples:run* compilation and execution of the examples of the library using the latest `Rust` version

4. doc
  - *pages* creation of documentation pages in HTML format and publishing through the GitLab pages service, this job is executed only on master branch

5. package
  - *package:build* verification of the creation of the library package

6. publish
  - *publish:send* creation of publishing on the public registry of the library package, this job is executed only on master branch and its activation is manual
