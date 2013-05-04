/* C Example */
#include <stdio.h>
#include <mpi.h>
#include <math.h>
#include <assert.h>
#include <stdlib.h>
#include <sys/time.h>

#define THRESHOLD 1
#define MAX_ITERS 20
#define MAX_DIM 128
#define CENTROIDS_TAG 1
#define PARTIAL_CENTROIDS_TAG 2
#define DONE_TAG 3
#define SQUARE(x,y) (x - y)*(x - y)

//#define DEBUG 1

typedef struct {
	char coordinates[MAX_DIM]; //Using max dim for convenience
	int num_points;
	int counts[MAX_DIM][4];
} Centroid; 

typedef struct {
	char *coordinates;
	int my_centroid;
} Point;

MPI_Datatype mpi_centroid_type = MPI_DATATYPE_NULL;

int iters;

static void init_mpi_centroid_type();
Point *read_points(char* data_file, int dims, long total_points, long* num_points_read);
void initialize_centroids_from_file(char *base_file, Centroid * centroids, int k,  int dims);
void initialize_centroids_from_points(Point* my_points, long num_points_read, Centroid *cs, int k, int dims);
void compute_distances(Point *points, int num_points, Centroid *curr_centroids, Centroid *partial_centroids, int k, int dims);
int str_distance(char* seq1, char* seq2, int length);
//int edit_distance(char *s1, char *s2, int dims);

bool update_centroids(Centroid *curr_centroids, Centroid *partial_centroids, int k, int dims);
void broadcast_new_centroid(Centroid *curr_centroids, int k, int dims);
void printCentroids(Centroid *centroids, int k, int dims);
void printPoints(Point *points, int k, int dims, int num_points);
double get_current_time_ms();
int get_dna_base_index(char c);
char get_dna_base(int index);
int main (int argc, char *argv[]) {
  MPI_Init (&argc, &argv);	/* starts MPI */

  int rank, size;
  MPI_Comm_rank (MPI_COMM_WORLD, &rank);	/* get current process id */
  MPI_Comm_size (MPI_COMM_WORLD, &size);	/* get number of processes */

	init_mpi_centroid_type();
	
	char* data_file = argv[1];
	int k = atoi(argv[2]);
	int dims = atoi(argv[3]);
	long total_points = atol(argv[4]);
	iters = 0;

	//Partial centroids. 
	//The coordinated will contain the sum of all points that belongs to this centroid on this processor 
	Centroid *partial_centroids = (Centroid *) malloc(k * sizeof(Centroid));
	Point *my_points;


	//Read files from the relevant part of the datafile.
	long num_points_read = 0;
	my_points = read_points(data_file, dims, total_points, &num_points_read);
#ifdef DEBUG
  	printf( "Process %d read %ld points from data file\n", rank, num_points_read);	
#endif

	double start = get_current_time_ms();
	//Allocate buffer for current centroids.
	Centroid *curr_centroids= (Centroid *) malloc(k * sizeof(Centroid));
	//Initialize curr_centroids
	//The 0th proccessor initializes k random centroids.
	if(rank == 0) {
		initialize_centroids_from_points(my_points, num_points_read, 
																		curr_centroids, k, dims);
		/*initialize_centroids_from_file(data_file, 
                        curr_centroids, k, dims);*/
	}

  broadcast_new_centroid(curr_centroids, k, dims);
 
	bool converged = false;
	do {
		//Compute distances from centroids and partial centroid sums
		compute_distances(my_points, num_points_read, curr_centroids, partial_centroids, k, dims);	

		converged = update_centroids(curr_centroids, partial_centroids, k, dims);
		iters++;
		if(rank == 0) {
			printf("Iteration %d:  ", iters);
			printCentroids(curr_centroids,  k, dims);
		} else {
#ifdef DEBUG
		printCentroids(curr_centroids,  k, dims);
#endif
		}
	} while(!converged);	
#ifdef DEBUG
		//printPoints(my_points, k, dims, num_points_read);
#endif

	if(rank == 0) {
		double end = get_current_time_ms();
		printf("Total time %lf\n", end - start);
	}
  MPI_Finalize();

  return 0;
}

//Initialize MPI data type.
static void init_mpi_centroid_type() {
  MPI_Datatype field_types[] = {MPI_CHAR, MPI_INT, MPI_INT};
  int field_lengths[] = {MAX_DIM, 1, MAX_DIM*4 };
  MPI_Aint field_offsets[] = { 0, MAX_DIM, MAX_DIM + 4};
  size_t num_fields = 3;

  int err = MPI_Type_create_struct(num_fields,
                                   field_lengths,
                                   field_offsets,
                                   field_types,
                                   &mpi_centroid_type);

  assert(err == MPI_SUCCESS);
  err = MPI_Type_commit(&mpi_centroid_type);
  assert(err == MPI_SUCCESS);

  int mpi_struct_size;

  assert(MPI_SUCCESS == MPI_Type_size(mpi_centroid_type, &mpi_struct_size));
	printf("%d\n %d\n", mpi_struct_size, sizeof(Centroid));
  assert(mpi_struct_size == sizeof(Centroid));
}

//TODO: Convert this to seek or assume different points for each process
Point *read_points(char* data_file, int dims, long total_points, long* num_points_read) {
  int rank, size;
  MPI_Comm_rank (MPI_COMM_WORLD, &rank);  /* get current process id */
  MPI_Comm_size (MPI_COMM_WORLD, &size);  /* get number of processes */

	char line[MAX_DIM + 1];
	int line_num = 0;
	
	long points_per_proc = total_points / size;
	long start_line = rank * points_per_proc;
	long end_line = rank == size-1 ? total_points : start_line + points_per_proc;
#ifdef DEBUG
	printf("Proc %d points_per_proc %ld start_line %ld\n", rank, points_per_proc, start_line);
#endif 

	Point * points = (Point *) malloc((end_line - start_line) * sizeof(Point));
	for(int i = 0; i < end_line - start_line; i++) {
		points[i].coordinates = (char *) malloc((dims+1)*sizeof(char));
	}
	int cnt = 0;

	FILE *fr = fopen (data_file, "r");  /* open the file for reading */

	while(fgets(line, 1024, fr) != NULL) {
		if(line_num >= start_line && line_num < end_line) {
			strcpy(points[cnt].coordinates, line);
			points[cnt].coordinates[dims] = '\0';
			cnt++;
		}
		line_num++;
	}

	*num_points_read = cnt;

	fclose(fr);

	return points;
}

void initialize_centroids_from_points(Point* my_points, long num_points_read, 
										Centroid *curr_centroids, int k, int dims) {
	//Naive way of initializing .. just take first k points assuming data is shuffled
	if(k > num_points_read) {
		printf("Not enough points to initialize k ... reduce num processes\n");
		exit(0);
	}

	for(int i = 0; i < k; i++) {
		strcpy(curr_centroids[i].coordinates, my_points[i].coordinates);
	}
}


void initialize_centroids_from_file(char *base_file, Centroid * centroids, int k,  int dims) {
	char cent_file[200];
	strcpy(cent_file, base_file);
	strcat(cent_file, ".cent");
	FILE *fr = fopen (cent_file, "r");  /* open the file for reading */

	char line[MAX_DIM];
	int i = 0;
	while(fgets(line, 1024, fr) != NULL) {
		if(i == k) {
			break;
		}		
		strcpy(centroids[i].coordinates, line);
		centroids[i].coordinates[dims] = '\0';
		i++;
	}

	fclose(fr);
}

void compute_distances(Point *points, int num_points, Centroid *curr_centroids, Centroid *partial_centroids, int k, int dims) {	
  int rank, size;
  MPI_Comm_rank (MPI_COMM_WORLD, &rank);  /* get current process id */
  MPI_Comm_size (MPI_COMM_WORLD, &size);  /* get number of processes */

	//Initialize partial centroids.
	for(int i = 0; i < k; i++) {
		partial_centroids[i].num_points = 0;
		for(int d = 0; d < dims; d++) {
			partial_centroids[i].counts[d][0] = 0;
			partial_centroids[i].counts[d][1] = 0;
			partial_centroids[i].counts[d][2] = 0;
			partial_centroids[i].counts[d][3] = 0;
		}
	}
	
	//For each point i
	for(int i = 0; i < num_points; i++) {
		int min_centroid = -1;
		int min_dist = 1000000;	
		//For each centroid j
		for(int j = 0; j < k; j++) {
			int dist = str_distance(curr_centroids[j].coordinates, points[i].coordinates, dims);	
			if(dist < min_dist) {
				min_centroid = j;
				min_dist = dist;
			}	
		}
		
		//Update the point with the centroid it belongs to.
		points[i].my_centroid = min_centroid;
		
		//Update the partial_centroid
		partial_centroids[min_centroid].num_points++;
		for(int d = 0; d < dims; d++) {
			int base_index = get_dna_base_index(points[i].coordinates[d]);
			if(base_index == -1){
				printf("ERROR1 %c!\n", points[i].coordinates[d]);
			}		
			partial_centroids[min_centroid].counts[d][base_index]++;
		}

	}

}

int str_distance(char* seq1, char* seq2, int length) {
	int similarity = 0;
	for(int i = 0; i < length; i++){
		if(*(seq1++) == *(seq2++)){
			similarity++;
		}
	}
	return (length - similarity);
}

/*
int edit_distance(char *s1, char *s2, int dims) {
}
*/

/**
 * Compute new centroids. Returns a bool variable indicating whether 
 * the algorithm has converged.
 */
bool update_centroids(Centroid *curr_centroids, Centroid *partial_centroids, int k, int dims) {
	int rank, size;
  MPI_Comm_rank (MPI_COMM_WORLD, &rank);  /* get current process id */
  MPI_Comm_size (MPI_COMM_WORLD, &size);  /* get number of processes */

	bool converged = false;
	
	if(rank == 0) {
		Centroid *tmp = (Centroid *) malloc(k * sizeof(Centroid));

		//Collect data for new centroids
		for(int proc = 1; proc < size; proc++) {
			MPI_Status status;
			//Receive new centroid from ith processor
			MPI_Recv(tmp, k, mpi_centroid_type, proc, PARTIAL_CENTROIDS_TAG,
            MPI_COMM_WORLD, &status);
		
			//Add proc processor's contribution towards centroid.
			for(int j = 0; j < k; j++) {
				partial_centroids[j].num_points += tmp[j].num_points;
				for(int d = 0; d < dims; d++) {
					for(int b = 0; b < 4; b++) {		
						partial_centroids[j].counts[d][b] += tmp[j].counts[d][b];
					}
				}
			}
		}
		free(tmp);
		
		//Compute new centroids.
		//Once all the centroids are received, compute the new centroid.
		int error = 0;
		for(int i = 0; i < k; i++) {
			if(partial_centroids[i].num_points == 0) {
				printf("Centroid %d has 0 points\n", i);
				continue;
			}
			for(int d = 0; d < dims; d++) {
				int max_base_index = -1;
				int max_base_count = -1;
				for(int b =0; b < 4; b++) {
					int count = partial_centroids[i].counts[d][b];
					if(max_base_count < count) {
						max_base_index = b;
						max_base_count = count;
					}
				}

				char new_base = get_dna_base(max_base_index);
				int new_base_index = max_base_index;
				int new_base_count = max_base_count;

				char old_base = curr_centroids[i].coordinates[d];
				int old_base_index = get_dna_base_index(old_base);
				int old_base_count = curr_centroids[i].counts[d][old_base_index];
						
				if(old_base != new_base) {
					error += old_base_count + new_base_count;
					/*printf("Old base index=%d Old Base count=%d new base index=%d new base count=%d\n", 
										old_base_index, old_base_count, max_base_index, max_base_count);*/
				} else {
					if(abs(old_base_count - new_base_count)) {
						/*printf("COUNTS DIFFER Old base=%c Old Base count=%d new base=%c new base count=%d\n", 
										old_base, old_base_count, new_base, new_base_count);*/
					}
			
					error += abs(old_base_count - new_base_count);
				}
				
				for(int b = 0; b < 4; b++) {		
					//Reset current centroid count.
					curr_centroids[i].counts[d][b] = 0;
				}
				curr_centroids[i].coordinates[d] = new_base;
				curr_centroids[i].counts[d][get_dna_base_index(new_base)] = new_base_count;
			}
		}

		printf("error %d\n", error);
		if(error < THRESHOLD || iters >= MAX_ITERS) {
		//if(iters >= MAX_ITERS) {
			converged = true;
		}
		int tag = converged ? DONE_TAG : CENTROIDS_TAG;
		//Send new centroids to everyone. 
		for(int proc = 1; proc < size; proc++) {
      	MPI_Send(curr_centroids, k, mpi_centroid_type, proc, tag, MPI_COMM_WORLD);
    }

	} else {
		MPI_Status status;
		//Send my contribution to processor 0.
		MPI_Send(partial_centroids, k, mpi_centroid_type, 0, PARTIAL_CENTROIDS_TAG,
            MPI_COMM_WORLD);
		
		//Receive the updated centroids from processor 0.
		MPI_Recv(curr_centroids, k, mpi_centroid_type, 0, MPI_ANY_TAG,
            MPI_COMM_WORLD, &status);
		//printf("status tag %d\n", status.MPI_TAG);
		if(status.MPI_TAG == DONE_TAG) {
			converged = true;	
		}
  
	}

	return converged;
}

void broadcast_new_centroid(Centroid *curr_centroids, int k, int dims) {	
	int rank, size;
  MPI_Comm_rank (MPI_COMM_WORLD, &rank);  /* get current process id */
  MPI_Comm_size (MPI_COMM_WORLD, &size);  /* get number of processes */
	
	if(rank == 0) {
		//Send initialize centriods to all other processors
		for(int i = 1; i < size; i++) {
		 	MPI_Send(curr_centroids, k, mpi_centroid_type, i, CENTROIDS_TAG,
            MPI_COMM_WORLD);
		}
	} else {
		MPI_Status status;
		//receive centroids from all other processors.
		MPI_Recv(curr_centroids, k, mpi_centroid_type, 0, CENTROIDS_TAG,
              MPI_COMM_WORLD, &status);
	}
}

void printCentroids(Centroid *centroids, int k, int dims) {	
	int rank, size;
  MPI_Comm_rank (MPI_COMM_WORLD, &rank);  /* get current process id */
  MPI_Comm_size (MPI_COMM_WORLD, &size);  /* get number of processes */	

	sleep(rank);

	printf("Proc %d: ", rank);
	for(int i = 0; i < k; i++) {
		printf(" C%d=(%s) ", i, centroids[i].coordinates);
	}
	printf("\n");
} 

void printPoints(Point *points, int k, int dims, int num_points) {
	int rank, size;
  MPI_Comm_rank (MPI_COMM_WORLD, &rank);  /* get current process id */
  MPI_Comm_size (MPI_COMM_WORLD, &size);  /* get number of processes */

	sleep(rank);

	for(int i = 0; i < num_points; i++){
		printf("Proc%d P%d: (%s)", rank, i, points[i]);	
	}

}

double get_current_time_ms() {
	struct timeval  tv;
	gettimeofday(&tv, NULL);

	double time_in_mill = 
         (tv.tv_sec) * 1000 + (tv.tv_usec) / (double)1000 ;
	return time_in_mill;
}

int get_dna_base_index(char c) {
	if(c == 'a') {
		return 0;
	} else if(c == 'c') {
		return 1;
	} else if(c == 'g') {
		return 2;
	} else if(c == 't') {
		return 3;
	} else {
		//printf("ERROR in getting index for character %c!\n", c);
		return -1;
	}
}

char get_dna_base(int index) {
	if(index == 0) {
		return 'a';
	} else if(index == 1) {
		return 'c';
	} else if(index == 2) {
		return 'g';
	} else if(index == 3) {
		return 't';
	} else {
		printf("ERROR in getting base for index %d", index);
		return -1;
	}
}
