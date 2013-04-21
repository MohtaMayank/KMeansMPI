#include <math.h>
#include <assert.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>

int main(int argc, char* argv[]) {
	

	char line[1024];
  int line_num = 0;

	char* data_file = argv[1];
	int dims = 2;

  int cnt = 0;

  FILE *fr = fopen (data_file, "r");  /* open the file for reading */

  while(fgets(line, 1024, fr) != NULL) {
      int dim = 0;
      char *pch = strtok(line," ");
      while (pch != NULL) {
        printf("%.3f,", atof(pch));
        pch = strtok(NULL, " ");
      }
			printf("\n");

      cnt++;
  }
	return 0;
}
