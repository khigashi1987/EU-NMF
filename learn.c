/*
 * learn.c
 */
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>
#include "MT.h"
#include "learn.h"
#include "feature.h"

void nmf_learn(double **data, int n_rows, int n_cols, int n_class, double **W, double **H, int maxiter){
    double **X_hat;
    int i,j;
    
    X_hat = (double **)calloc(n_rows,sizeof(double *));
    for(i = 0;i < n_rows;i++){
        X_hat[i] = (double *)calloc(n_cols,sizeof(double));
    }
    
    // initialize W, H
    init_genrand(time(NULL));
    // W(n_rows, n_class)
    for(i = 0;i < n_rows;i++){
        for(j = 0;j < n_class;j++){
            W[i][j] = genrand_real3();
        }
    }
    // H(n_class, n_cols)
    for(i = 0;i < n_class;i++){
        for(j = 0;j < n_cols;j++){
            H[i][j] = genrand_real3();
        }
    }
    
    // X_hat = W x H
    int k;
    for(i = 0;i < n_rows;i++){
        for(j = 0;j < n_cols;j++){
            X_hat[i][j] = 0.0;
            for(k = 0;k < n_class;k++){
                X_hat[i][j] += W[i][k] * H[k][j];
            }
        }
    }
    
    
    FILE *ofp;
    if((ofp = fopen("EU2.txt","w")) == NULL){
        fprintf(stderr,"nmf_learn:: cannot open output file.\n");
        exit(1);
    }
    fprintf(ofp,"STEP\tEU2\n");
    
    
    // iteration
    int it;
    double numerator;
    double denominator;
    double eu;
    double prev_eu;
    double converge_threshold = 1.0e-12;
    double epsilon = 1.0e-12;
    for(it = 0;it < maxiter;it++){
        printf("iteration %2d / %3d..\n",it+1,maxiter);
        fflush(stdout);
        // update rules for minimizing IS divergence
        // update W
        for(i = 0;i < n_rows;i++){
            for(k = 0;k < n_class;k++){
                if(W[i][k] != 0.0){
                    numerator = 0.0;
                    denominator = 0.0;
                    for(j = 0;j < n_cols;j++){
                        numerator += data[i][j] * H[k][j];
                        denominator += X_hat[i][j] * H[k][j];
                    }
                    if(denominator != 0.0){
                        W[i][k] = W[i][k] * (numerator / denominator);
                    }else{
                        W[i][k] = W[i][k] * (numerator / epsilon);
                    }
                }
            }
        }
        // update X_hat
        for(i = 0;i < n_rows;i++){
            for(j = 0;j < n_cols;j++){
                X_hat[i][j] = 0.0;
                for(k = 0;k < n_class;k++){
                    X_hat[i][j] += W[i][k] * H[k][j];
                }
            }
        }
        // update H
        for(k = 0;k < n_class;k++){
            for(j = 0;j < n_cols;j++){
                if(H[k][j] != 0.0){
                    numerator = 0.0;
                    denominator = 0.0;
                    for(i = 0;i < n_rows;i++){
                        numerator += data[i][j] * W[i][k];
                        denominator += X_hat[i][j] * W[i][k];
                    }
                    if(denominator != 0.0){
                        H[k][j] = H[k][j] * (numerator / denominator);
                    }else{
                        H[k][j] = H[k][j] * (numerator / epsilon);
                    }
                }
            }
        }
        // update X_hat
        for(i = 0;i < n_rows;i++){
            for(j = 0;j < n_cols;j++){
                X_hat[i][j] = 0.0;
                for(k = 0;k < n_class;k++){
                    X_hat[i][j] += W[i][k] * H[k][j];
                }
            }
        }
        // compute Eu^2
        eu = 0.0;
        for(i = 0;i < n_rows;i++){
            for(j = 0;j < n_cols;j++){
                eu += (data[i][j] - X_hat[i][j]) * (data[i][j] - X_hat[i][j]);
            }
        }
        fprintf(stdout,"\n Euclid Distance ^ 2 = %.8f\n",eu);
        fprintf(ofp,"%d\t%.8f\n",it,eu);
        if((it != 0) && (fabs(prev_eu - eu) < converge_threshold)){
            printf("converged.\n");
            break;
        }
        prev_eu = eu;
    }
    fclose(ofp);
}
