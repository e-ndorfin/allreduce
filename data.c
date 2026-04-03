/*
 * Data loading for semeion.data training set.
 *
 * The lns() and readln() functions below were taken from test.c in the
 * Tinn library (https://github.com/glouw/tinn). The parse_row() and
 * data_load() functions are our own, storing data in a flat float array
 * for pipe transmission rather than Tinn's original 2D array format.
 */

#include "data.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

// Returns the number of lines in a file. (Adapted from Tinn test.c)
static int lns(FILE *file)
{
    int ch = EOF;
    int lines = 0;
    int pc = '\n';
    while ((ch = getc(file)) != EOF)
    {
        if (ch == '\n')
            lines++;
        pc = ch;
    }
    if (pc != '\n')
        lines++;
    rewind(file);
    return lines;
}

// Reads a line from a file. (Adapted from Tinn test.c)
static char *readln(FILE *file)
{
    int ch = EOF;
    int reads = 0;
    int size = 128;
    char *line = (char *)malloc(size * sizeof(char));
    while ((ch = getc(file)) != '\n' && ch != EOF)
    {
        line[reads++] = ch;
        if (reads + 1 == size)
            line = (char *)realloc(line, (size *= 2) * sizeof(char));
    }
    line[reads] = '\0';
    return line;
}

// Parses one row of space-separated floats into the flat array.
static void parse_row(float *row, char *line, int cols)
{
    for (int col = 0; col < cols; col++)
    {
        const float val = atof(strtok(col == 0 ? line : NULL, " "));
        row[col] = val;
    }
}

Data data_load(const char *path, int nips, int nops)
{
    FILE *file = fopen(path, "r");
    if (file == NULL)
    {
        fprintf(stderr, "Could not open %s\n", path);
        exit(1);
    }
    const int rows = lns(file);
    const int cols = nips + nops;
    Data data;
    data.nips = nips;
    data.nops = nops;
    data.rows = rows;
    data.flat = (float *)malloc(rows * cols * sizeof(float));
    for (int row = 0; row < rows; row++)
    {
        char *line = readln(file);
        parse_row(data.flat + row * cols, line, cols);
        free(line);
    }
    fclose(file);
    return data;
}

void data_free(Data d)
{
    free(d.flat);
}
