# Python Tools

Collection of small Python Programs.
Programs should have a cli and a Django Web interface.

## Programs

### PDF

Todo: Rename

Count Word Frequency of one or multiple documents and optionally compares them against each other.
Defines Computation Stages which may store intermediary results which take long time to compute.

Currently defines 3 Stages:

1. Extract
    - Extracts the Text Portion of the Document as good as possible
    - Extracted Text is stored
2. Process
    - Process the Text of the document and computes the word frequencies and TD-IF Values
3. Compare
    - Computes the similarity between each document (currently the cosine similarity via TD-IF values)

#### CLI
Not available at the moment.

#### Django Web Interface

Should allow the Customization of various Input Parameter:

    processes: int - number of processes each stage should use

Displays the current Progress of a Execution via a Task Object.

## Server

The Server allows access to multiple Components, where most describe a Program (Python Tool) or a Result Viewer.

### Current Components

#### Library

View the Results of the Documents

#### Library Calculator

The Web Interface for the '**PDF**' Program.