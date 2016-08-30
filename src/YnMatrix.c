//	File        :   YnMatrix.c
//	Brief       :   Implement methods.
//	DD-MM-YYYY  :   26-07-2016
//	Author      :   haittt

#include "../include/YnMatrix.h"

/**************** Define */

/**************** Macro */

/**************** Enum */

/**************** Struct */

/**************** Local variables */

/**************** Global variables */

/**************** Local Implement */

/**************** Implement */
void YnMatrixFree(tYnMatrix matrix)
{
    uint32 i;
    for (i = 0; i < matrix.rows; i ++)
        YnUtilFree(matrix.vals[i]);

    YnUtilFree(matrix.vals);
}

float YnMatrixTopAccuracy(tYnMatrix truth,
        tYnMatrix guess,
        int k)
{
    int *indexes = calloc(k, sizeof(int));
    int n = truth.cols;
    int i,j;
    int class;
    int correct = 0;

    for (i = 0; i < truth.rows; i ++)
    {
        YnUtilTop(guess.vals[i], n, k, indexes);

        for (j = 0; j < k; j ++)
        {
            class = indexes[j];
            if (truth.vals[i][class])
            {
                ++correct;
                break;
            }
        }
    }

    YnUtilFree(indexes);

    return (float)(correct / truth.rows);
}

void YnMatrixAddMatrix(tYnMatrix from,
        tYnMatrix to)
{
    int i,j;

    assert((from.rows == to.rows) && (from.cols == to.cols));

    for (i = 0; i < from.rows; i ++)
    {
        for (j = 0; j < from.cols; j ++)
        {
            to.vals[i][j] += from.vals[i][j];
        }
    }
}

tYnMatrix YnMatrixMake(uint32 rows,
        uint32 cols)
{
    int i;
    tYnMatrix matrix;

    matrix.rows = rows;
    matrix.cols = cols;
    matrix.vals = calloc(matrix.rows, sizeof(float *));

    for (i = 0; i < matrix.rows; i ++)
    {
        matrix.vals[i] = calloc(matrix.cols, sizeof(float));
    }

    return matrix;
}

tYnMatrix YnMatrixHoldOut(tYnMatrix * matrix,
        int n)
{
    int i;
    int index;
    tYnMatrix hold;

    hold.rows = n;
    hold.cols = matrix->cols;
    hold.vals = calloc(hold.rows, sizeof(float *));

    for (i = 0; i < n; i ++)
    {
        index = rand() % (matrix->rows);
        hold.vals[i] = matrix->vals[index];
        matrix->vals[index] = matrix->vals[--(matrix->rows)];
    }

    return hold;
}

float * YnMatrixPopColumn(tYnMatrix * matrix,
        int c)
{
    float *col = calloc(matrix->rows, sizeof(float));
    int i, j;

    for (i = 0; i < matrix->rows; i ++)
    {
        col[i] = matrix->vals[i][c];

        for (j = c; j < (matrix->cols - 1); j ++)
        {
            matrix->vals[i][j] = matrix->vals[i][j+1];
        }
    }

    -- matrix->cols;

    return col;
}

tYnMatrix YnMatrixCsvToMatrix(char *filename)

{
    char *line;
    tYnMatrix matrix;
    FILE *fp = fopen(filename, "r");
    int n = 0;
    int size = 1024;

    if (!fp)
        YnUtilErrorOpenFile(filename);

    matrix.cols = -1;
    matrix.vals = calloc(size, sizeof(float*));\

    while ((line = fgetl(fp)))
    {
        if (matrix.cols == -1)
            matrix.cols = count_fields(line);

        if (n == size)
        {
            size *= 2;
            matrix.vals = realloc(matrix.vals, size * sizeof(float*));
        }

        matrix.vals[n] = YnUtilLineFieldParse(line, matrix.cols);
        YnMatrixFree(line);

        ++ n;
    }

    matrix.vals = realloc(matrix.vals, n * sizeof(float*));
    matrix.rows = n;

    return matrix;
}

void YnMatrixPrint(tYnMatrix matrix)
{
    int i, j;

    printf("%d X %d Matrix:\n", matrix.rows, matrix.cols);
    printf(" __");

    for (j = 0; j < ((16 * matrix.cols) - 1); j ++)
        printf(" ");
    printf("__ \n");

    printf("|  ");
    for (j = 0; j < ((16 * matrix.cols) - 1); j ++)
        printf(" ");
    printf("  |\n");

    for (i = 0; i < matrix.rows; i ++)
    {
        printf("|  ");

        for (j = 0; j < matrix.cols; j ++)
        {
            printf("%15.7f ", matrix.vals[i][j]);
        }
        printf(" |\n");
    }

    printf("|__");
    for (j = 0; j < ((16 * matrix.cols) - 1); j ++)
        printf(" ");
    printf("__|\n");
}
