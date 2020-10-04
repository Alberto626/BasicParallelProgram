#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <mpi.h> 

void Read_n(int* n_p, int* local_n_p, int my_rank, int comm_sz, 
      MPI_Comm comm);
void Check_for_error(int local_ok, char fname[], char message[], 
      MPI_Comm comm);
void Read_data(double local_vec1[], double local_vec2[], double* scalar_p,
      int local_n, int my_rank, int comm_sz, MPI_Comm comm);
void Print_vector(double local_vec[], int local_n, int n, char title[], 
      int my_rank, MPI_Comm comm);
double Par_dot_product(double local_vec1[], double local_vec2[], 
      int local_n, MPI_Comm comm);
void Par_vector_scalar_mult(double local_vec[], double scalar, 
      double local_result[], int local_n);

int main(void) {
   int n, local_n;
   double *local_vec1, *local_vec2;
   double *local_scalar_mult1, *local_scalar_mult2;
   double scalar;
   double dot_product;
   int comm_sz, my_rank;
   MPI_Init(NULL,NULL);
   MPI_Comm_size(MPI_COMM_WORLD, &comm_sz); /*set comm_sz to the number of processesors*/
   MPI_Comm_rank(MPI_COMM_WORLD, &my_rank); /*set my _rank to the process number*/
   MPI_Comm comm = MPI_COMM_WORLD;
   
   /* Print input data */
   Read_n(&n, &local_n, my_rank, comm_sz, comm);
   local_vec1 = malloc(local_n * sizeof(double));
   local_vec2 = malloc(local_n * sizeof(double));
   Read_data(local_vec1,local_vec2, &scalar, local_n, my_rank, comm_sz, comm);
   /* Print results */
   
   /* Compute and print dot product */
   double par_dot = Par_dot_product(local_vec1, local_vec2, local_n,comm);
   printf("Processor %d with part dot equals %lf \n", my_rank, par_dot);
   double* dot = NULL;
   if(my_rank == 0) {
         dot = malloc(comm_sz * sizeof(double));
   }
   MPI_Gather(&par_dot, 1, MPI_DOUBLE, dot, 1, MPI_DOUBLE, 0, comm);
   if(my_rank == 0) {
         int i;
         dot_product = 0;
         for(i = 0; i < comm_sz; i++) {
               dot_product += dot[i];
         }
         printf("Total Dot Product is: %lf\n", dot_product);
   }
   
   /* Compute scalar multiplication and print out result */
   local_scalar_mult1 = malloc(local_n * sizeof(double));
   local_scalar_mult2 = malloc(local_n * sizeof(double));
   double* scalar_mult1;
   double* scalar_mult2;
   Par_vector_scalar_mult(local_vec1,scalar, local_scalar_mult1, local_n);
   Par_vector_scalar_mult(local_vec2,scalar, local_scalar_mult2, local_n);
   if (my_rank == 0) {
         scalar_mult1 = malloc(n * sizeof(double));
         scalar_mult2 = malloc(n * sizeof(double));

   }
   MPI_Gather(local_scalar_mult1, local_n, MPI_DOUBLE, scalar_mult1, local_n, MPI_DOUBLE, 0, comm);
   MPI_Gather(local_scalar_mult2, local_n, MPI_DOUBLE, scalar_mult2, local_n, MPI_DOUBLE, 0, comm);
   if(my_rank == 0) {
         printf("Scalared Vector 1 at %d is: ", my_rank);
         int i;
         for(i = 0; i < n; i++) {
               printf("%lf ", scalar_mult1[i]);
         }
         printf("\n");
          printf("Scalared Vector 2 at %d is: ", my_rank);
         for(i = 0; i < n; i++) {
               printf("%lf ", scalar_mult2[i]);
         }
         printf("\n");
   }
   
   
   free(local_vec2);
   free(local_vec1);
   
   MPI_Finalize();
   return 0;
}

/*-------------------------------------------------------------------*/
void Check_for_error(
                int       local_ok   /* in */, 
                char      fname[]    /* in */,
                char      message[]  /* in */, 
                MPI_Comm  comm       /* in */) {
   int ok;
   
   MPI_Allreduce(&local_ok, &ok, 1, MPI_INT, MPI_MIN, comm);
   if (ok == 0) {
      int my_rank;
      MPI_Comm_rank(comm, &my_rank);
      if (my_rank == 0) {
         fprintf(stderr, "Proc %d > In %s, %s\n", my_rank, fname, 
               message);
         fflush(stderr);
      }
      MPI_Finalize();
      exit(-1);
   }
}  /* Check_for_error */


/* Get the input of n: size of the vectors, and then calculate local_n according to comm_sz and n */
/* where local_n is the number of elements each process obtains */
/*-------------------------------------------------------------------*/
void Read_n(int* n_p, int* local_n_p, int my_rank, int comm_sz, 
      MPI_Comm comm) {
      if(my_rank == 0) {
            printf("Input the size of the vectors\n");
            scanf("%d",n_p); /*set the size of the vecotr*/
            printf("Vector size is %d \n", *n_p);
            *local_n_p = *n_p / comm_sz; /*set the size of each processor to get*/
      }
     MPI_Bcast(local_n_p, 1, MPI_INT, 0, comm);

}  /* Read_n */


/* local_vec1 and local_vec2 are the two local vectors of size local_n which the process pertains */
/* process 0 will take the input of the scalar, the two vectors a and b */
/* process 0 will scatter the two vectors a and b across all processes */
/*-------------------------------------------------------------------*/
void Read_data(double local_vec1[], double local_vec2[], double* scalar_p,
      int local_n, int my_rank, int comm_sz, MPI_Comm comm) {
   double* a = NULL; /*the total first vector*/
   double* b = NULL; /*the total second vector*/
   int i;
   int n = local_n * comm_sz;
   if (my_rank == 0){
      printf("What is the scalar?\n");
      scanf("%lf", scalar_p);
   }
   
   MPI_Bcast(scalar_p, 1, MPI_DOUBLE, 0, comm); /*Send to all processors that scalar_p equals __*/
   
   if (my_rank == 0){
      a = malloc(n * sizeof(double));
      printf("Enter the first vector\n");
      for (i = 0; i < n; i++) {
         scanf("%lf", &a[i]);
      }
      b = malloc(n * sizeof(double));
      printf("Enter the second vector\n");
      for (i = 0; i < n; i++) {
         scanf("%lf", &b[i]);
      }
      }
      MPI_Scatter(a, local_n, MPI_DOUBLE, local_vec1, local_n, MPI_DOUBLE, 0, comm);
      MPI_Scatter(b, local_n, MPI_DOUBLE, local_vec2, local_n, MPI_DOUBLE, 0, comm);
      Print_vector(local_vec1,local_n, n, "", my_rank, comm);
      Print_vector(local_vec2,local_n, n, "", my_rank, comm);


}  /* Read_data */

/* The print_vector gathers the local vectors from all processes and print the gathered vector */
/*-------------------------------------------------------------------*/
void Print_vector(double local_vec[], int local_n, int n, char title[], 
      int my_rank, MPI_Comm comm) {
   double* a = NULL;
   int i;
   
   if (my_rank == 0) {
      a = malloc(n * sizeof(double));
   }
   MPI_Gather(local_vec, local_n, MPI_DOUBLE, a, local_n, MPI_DOUBLE, 0, comm);
   if(my_rank == 0) {
         printf("Total vector is of processor: %d", my_rank);
         for (i = 0; i < n; i++) {
             printf(": %lf", a[i]);
         }
         printf("\n");
         
         free(a);
   } 
   printf("process %d holds:", my_rank);
   for (i = 0; i < local_n; i++) /*print local */{
       printf("%lf ", local_vec[i]);
   }
   printf("\n");
   

}  /* Print_vector */


/* This function computes and returns the partial dot product of local_vec1 and local_vec2 */
/*-------------------------------------------------------------------*/
double Par_dot_product(double local_vec1[], double local_vec2[], 
      int local_n, MPI_Comm comm) {
	int i;
	double count =0;
	for(i = 0; i < local_n; i++) {
		count += (local_vec1[i] * local_vec2[i]);
      }
      return count;
}  /* Par_dot_product */


/* This function gets the vector which is the scalar times local_vec, and put the vector into local_result */
/*-------------------------------------------------------------------*/
void Par_vector_scalar_mult(double local_vec[], double scalar, 
      double local_result[], int local_n) {
            int x;
            for (x = 0; x < local_n; x++)
            {
                  local_result[x] = local_vec[x] * scalar;
            }
}  /* Par_vector_scalar_mult */
