#ifndef DATA_H
#define DATA_H

typedef struct
{
    // Flat array: rows * (nips + nops) floats.
    // Each row is [inputs... targets...].
    float *flat;
    int nips;
    int nops;
    int rows;
}
Data;

Data data_load(const char *path, int nips, int nops);

void data_free(Data d);

#endif
