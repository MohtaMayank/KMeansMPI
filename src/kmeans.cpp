/* C Example */
#include <stdio.h>
#include <mpi.h>
#include <math.h>
#include <assert.h>
#include <stdlib.h>
#include <sys/time.h>

#define THRESHOLD 0.001
#define MAX_ITERS 20
#define MAX_DIM 128
#define CENTROIDS_TAG 1
#define PARTIAL_CENTROIDS_TAG 2
#define DONE_TAG 3
#define SQUARE(x,y) (x - y)*(x - y)

//#define DEBUG 1

typedef struct {
	int num_points;
	float coordinates[MAX_DIM];	//Fixed upper limit of dimensions for easy transfer with MPI.
} Centroid; 

typedef struct {
	float *coordinates;
	int my_centroid;
} Point;

MPI_Datatype mpi_centroid_type = MPI_DATATYPE_NULL;

int iters;

static void init_mpi_centroid_type();
Point *read_points(char* data_file, int dims, long total_points, long* num_points_read);
void initialize_centroids(Centroid * centroids, int k,  int dims);
void initialize_centroids_from_points(Point* my_points, long num_points_read, Centroid *cs, int k, int dims);
void compute_distances(Point *points, int num_points, Centroid *curr_centroids, Centroid *partial_centroids, int k, int dims);
bool update_centroids(Centroid *curr_centroids, Centroid *partial_centroids, int k, int dims);
void broadcast_new_centroid(Centroid *curr_centroids, int k, int dims);
void printCentroids(Centroid *centroids, int k, int dims);
void printPoints(Point *points, int k, int dims, int num_points);
double get_current_time_ms();

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
	for(int i = 0; i < k; i++) {
		curr_centroids[i].num_points = 1;
		//The 0th proccessor initializes k random centroids.
		if(rank == 0) {
			//initialize_centroids(curr_centroids, k, dims);
			initialize_centroids_from_points(my_points, num_points_read, 
												curr_centroids, k, dims);
#ifdef DEBUG
			printf("centroids initialized\n");
#endif
		}
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
  MPI_Datatype field_types[] = {MPI_INT, MPI_FLOAT };
  int field_lengths[] = {1, MAX_DIM };
  MPI_Aint field_offsets[] = { 0, 1 };
  size_t num_fields = 2;

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
  assert(mpi_struct_size == sizeof(Centroid));
}

//TODO: Convert this to seek or assume different points for each process
Point *read_points(char* data_file, int dims, long total_points, long* num_points_read) {
  int rank, size;
  MPI_Comm_rank (MPI_COMM_WORLD, &rank);  /* get current process id */
  MPI_Comm_size (MPI_COMM_WORLD, &size);  /* get number of processes */

	char line[1024];
	int line_num = 0;
	
	long points_per_proc = total_points / size;
	long start_line = rank * points_per_proc;
	long end_line = rank == size-1 ? total_points : start_line + points_per_proc;
#ifdef DEBUG
	printf("Proc %d points_per_proc %ld start_line %ld\n", rank, points_per_proc, start_line);
#endif 

	Point * points = (Point *) malloc((end_line - start_line) * sizeof(Point));
	for(int i = 0; i < end_line - start_line; i++) {
		points[i].coordinates = (float *) malloc(dims*sizeof(float));
	}
	int cnt = 0;

	FILE *fr = fopen (data_file, "r");  /* open the file for reading */

	while(fgets(line, 1024, fr) != NULL) {
		if(line_num >= start_line && line_num < end_line) {
			int dim = 0;
			char *pch = strtok (line," ");
  		while (pch != NULL) {
				points[cnt].coordinates[dim++] = atof(pch);
   			pch = strtok (NULL, " ");
  		}
			if(dim != dims) {
				printf("\nERROR! not enough dimension ... exitting\n");
				exit(2);
			}
			
			cnt++;
		}
		line_num++;
	}

	*num_points_read = cnt;

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
		for(int d = 0; d < dims; d++) {
			curr_centroids[i].coordinates[d] = my_points[i].coordinates[d];
		}
	}
}

void initialize_centroids(Centroid * curr_centroids, int k, int dims) {
		//Initialize the centroids - Assuming all points are between 0 and 1
		srand((unsigned)time(NULL));
		for(int i = 0; i < k; i++) {
			for(int j = 0; j < dims; j++) {
				curr_centroids[i].coordinates[j] = ((float)rand()/(float)RAND_MAX);
			}
		}
}

void compute_distances(Point *points, int num_points, Centroid *curr_centroids, Centroid *partial_centroids, int k, int dims) {	
  int rank, size;
  MPI_Comm_rank (MPI_COMM_WORLD, &rank);  /* get current process id */
  MPI_Comm_size (MPI_COMM_WORLD, &size);  /* get number of processes */

	//Initialize partial centroids.
	for(int i = 0; i < k; i++) {
		partial_centroids[i].num_points = 0;
		for(int d = 0; d < dims; d++) {
			partial_centroids[i].coordinates[d] = 0;
		}
	}
	
	//For each point i
	for(int i = 0; i < num_points; i++) {
		int min_centroid = -1;
		float min_dist = 1000000;	
		//For each centroid j
		for(int j = 0; j < k; j++) {
			float dist = 0;
			for(int d = 0; d < dims; d++) {		
				dist += SQUARE(points[i].coordinates[d], curr_centroids[j].coordinates[d]);
			}
			
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
			partial_centroids[min_centroid].coordinates[d] += points[i].coordinates[d];
		}

	}

}


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
					partial_centroids[j].coordinates[d] += tmp[j].coordinates[d];
				}
			}
		}
		free(tmp);
		
		//Compute new centroids.
		//Once all the centroids are received, compute the new centroid.
		float error = 0;
		for(int i = 0; i < k; i++) {
			if(partial_centroids[i].num_points == 0) {
				printf("Centroid %d has 0 points\n", i);
				continue;
			}
			for(int d = 0; d < dims; d++) {
				float new_coord = (partial_centroids[i].coordinates[d] / (float) partial_centroids[i].num_points);
				error = error + fabs(new_coord - curr_centroids[i].coordinates[d]);
				curr_centroids[i].coordinates[d] = new_coord;
			}
		}

		printf("error %f\n", error);
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
#ifdef DEBUG
		printCentroids(curr_centroids,  k, dims);
#endif
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
#ifdef DEBUG
		printCentroids(curr_centroids, k, dims);
#endif
	}
}

void printCentroids(Centroid *centroids, int k, int dims) {	
	int rank, size;
  MPI_Comm_rank (MPI_COMM_WORLD, &rank);  /* get current process id */
  MPI_Comm_size (MPI_COMM_WORLD, &size);  /* get number of processes */	

	sleep(rank);

	printf("Proc %d: ", rank);
	for(int i = 0; i < k; i++) {
		printf(" C%d=(", i);
		for(int d = 0; d < dims; d++) {
			printf("%.3f,", centroids[i].coordinates[d]);
		}
		printf(") ");	
	}
	printf("\n");
} 

void printPoints(Point *points, int k, int dims, int num_points) {
	int rank, size;
  MPI_Comm_rank (MPI_COMM_WORLD, &rank);  /* get current process id */
  MPI_Comm_size (MPI_COMM_WORLD, &size);  /* get number of processes */

	sleep(rank);

	for(int i = 0; i < num_points; i++){
		printf("Proc%d P%d: (", rank, i);	
		for(int d = 0; d < dims; d++) {
			printf("%.3f ", points[i].coordinates[d]);
		}
		printf(") Centroid %d\n", points[i].my_centroid);	
	}

}

double get_current_time_ms() {
	struct timeval  tv;
	gettimeofday(&tv, NULL);

	double time_in_mill = 
         (tv.tv_sec) * 1000 + (tv.tv_usec) / (double)1000 ;
	return time_in_mill;
}
