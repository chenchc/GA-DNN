#include<stdio.h>
#include<iostream>
#include<math.h>
#include <vector>
#include<time.h>
#include <algorithm>
#include <cfloat>
#include "myrand.h"
#include <functional>
#include <omp.h>

using namespace std;

struct sol {
	double x;//accuracy
	int y;//cost
	int part;
	int age;
	bool isChoose = false;
	vector<bool>Chromosome;
};

void initial();
void selection();
int noneDominate( const sol &, const sol & );
void noneDominateSort( void );
void crowdingDisSort( void );
void crossover( void );
double getMSE( int, int );
void storeBest( int, int );

const int maxGeneration = 1000;
int inputNode;
int outputNode;
int length;
int populationSize;
const int nElder = 3;
const int elderAge = 5;
int edgePart;
int currentNElder = 0;
MyRand randGenerator;
char *jobname;
char *trainDataFilename;

vector<sol> solution;
double *crowdingDisPart;
vector<int> indexPart;
vector<int> firstList;
int *validList;
int *invalidList;
int validCount;
int curentGeneration = 0;

int main(int argc, char **argv) {
	if (argc != 5) {
		cerr << "Wrong parameters" << endl;
	}

	inputNode = atoi(argv[1]);
	outputNode = atoi(argv[2]);
	length = inputNode * outputNode;
	populationSize = (int)(length * log(length) / 64) / 2 * 2;
	jobname = argv[3];
	trainDataFilename = argv[4];
	crowdingDisPart = new double[populationSize * 2]{0.0};
	validList = new int[populationSize]{0};
	invalidList = new int[populationSize]{0};
	solution = vector<sol>(2 * populationSize);

	omp_set_num_threads(8);

	int i, j;
	FILE *fp = fopen( "spec", "w" );

	printf( "%s\n%d\n%d\n", trainDataFilename, inputNode, outputNode );
	fprintf( fp, "%s\n%d\n%d\n", trainDataFilename, inputNode, outputNode );
	fclose( fp );

	initial();
	while ( currentNElder < nElder && curentGeneration < maxGeneration ) {
		crossover();
		//getMSE();
		selection();
		for ( i = 0; i < populationSize; i++ ) {
			printf( "%d-%lf,%d,%d ", solution[validList[i]].part, solution[validList[i]].x, solution[validList[i]].y, solution[validList[i]].age );
		}
		curentGeneration++;
		printf( "\ngenration %d\nnum of first = %d\n\n", curentGeneration, indexPart.size() );
	}

	/*
	fp = fopen( "temp.txt", "w" );
	fprintf( fp, "X = [" );
	for ( j = 0; j < populationSize; j++ ) {
		fprintf( fp, "%d ", (int) solution[validList[j]].x );
	}
	fprintf( fp, "];\nY = [" );
	for ( j = 0; j < populationSize; j++ ) {
		fprintf( fp, "%d ", solution[validList[j]].y );
	}
	fprintf( fp, "];\ni = %d;\nscatter(-X(1:i),Y(1:i),'.');", k );
	fclose( fp );

	while ( 1 );
	*/
}
void initial() {
	int i = 0, j, weight = 0, cost = 0;

	for ( i = 0; i <populationSize; i++ ) {
		solution[i].age = 0;
		solution[i + populationSize].age = 0;
		validList[i] = i;
		invalidList[i] = i + populationSize;
		for ( j = 0; j < length; j++ ) {
			solution[i + populationSize].Chromosome.push_back( 0 );
			if ( randGenerator.uniform( 0.0, 1.0 ) > 0.75) {
				solution[i].Chromosome.push_back( 1 );
				cost++;
			} else {
				solution[i].Chromosome.push_back( 0 );
			}

		}
		solution[i].y = cost;
		cost = 0;
	}
	#pragma omp parallel for
	for (i = 0; i < populationSize; i++) {
		int tid = omp_get_thread_num();
		solution[i].x = getMSE(i, tid);
	}
}

void selection() {
	int i, j;
	bool invalidTable[2 * populationSize] = {0};

	noneDominateSort();
	crowdingDisSort();

	for ( i = 0; i < indexPart.size(); i++ ) {
		if ( validCount < populationSize ) {
			validList[validCount++] = indexPart[i];
			solution[indexPart[i]].age++;
		} else
			break;
	}
	// prepare indexPart to store first part
	indexPart.clear();
	currentNElder = 0;
	for ( i = 0; i < populationSize; i++ ) {
		invalidTable[validList[i]] = 1;
		// elder convergence
		if ( solution[validList[i]].age >= elderAge && solution[validList[i]].part == 1) {
			currentNElder++;
		}
		if ( solution[validList[i]].part == 1 ) {
			indexPart.push_back( validList[i] );
		}
	}
	crowdingDisSort();   // rearrange first part by its crowding distance
	for ( i = 0; i < nElder; i++ ) {
		storeBest( indexPart[i], i + 1 );
	}

	j = 0;
	for ( i = 0; i < 2 * populationSize; i++ ) {
		if ( !invalidTable[i] ) {
			invalidList[j++] = i;
			solution[i].age = 0;
		}
		// initialization for next generation
		solution[i].isChoose = false;
	}
}

void polymorphismModify(vector<bool> &parent1, vector<bool> &parent2) {
	const int sampleCount = inputNode / 5;

	for (int i = 0; i < outputNode; i++) {
		for (int j = 0; j < outputNode; j++) {
			if (i == j)
				continue;

			bool similar = true;
			for (int j = 0; j < sampleCount; j++) {
				int inputNodeIndex = randGenerator.uniformInt(0, inputNode - 1);
				if (parent1[i * inputNode + inputNodeIndex] !=
					parent2[j * inputNode + inputNodeIndex]) 
				{
					similar = false;	
				}
			}
			if (!similar)
				continue;
			// Swap
			vector<bool> temp(
				parent2.begin() + j * inputNode, 
				parent2.begin() + (j + 1) * inputNode);
			for (int k = 0; k < inputNode; k++)
				parent2[j * inputNode + k] = parent2[i * inputNode + k];
			for (int k = 0; k < inputNode; k++)
				parent2[i * inputNode + k] = temp[k];
		}
	}
}

/*
void polymorphismModify(vector<bool> &parent1, vector<bool> &parent2) {
	vector<size_t> hashOfHiddenParent1;
	vector<size_t> hashOfHiddenParent2;
	hash<vector<bool> > hash_edge;

	for (int i = 0; i < length; i += inputNode) {
		hashOfHiddenParent1.push_back(hash_edge(
			vector<bool>(parent1.begin() + i, parent1.begin() + i + inputNode)));
		hashOfHiddenParent2.push_back(hash_edge(
			vector<bool>(parent2.begin() + i, parent2.begin() + i + inputNode)));
	}

	for (int i = 0; i < outputNode; i++) {
		size_t hashValueParent1 = hashOfHiddenParent1[i];
		for (int j = 0; j < outputNode; j++) {
			size_t hashValueParent2 = hashOfHiddenParent2[j];
			if (i == j)
				continue;
			if (hashValueParent1 == hashValueParent2) {
				cout << "Swap!" << endl;
				vector<bool> temp(
					parent2.begin() + j * inputNode, 
					parent2.begin() + (j + 1) * inputNode);
				for (int k = 0; k < inputNode; k++)
					parent2[j * inputNode + k] = parent2[i * inputNode + k];
				for (int k = 0; k < inputNode; k++)
					parent2[i * inputNode + k] = temp[k];
			}
		}
	}
}
*/

void crossover() {
	int i, j, parent1, parent2, child1, child2;
	int weight1, weight2, cost1, cost2;
	int permutation[populationSize] = {0};

	randGenerator.uniformArray( permutation, populationSize, 0, populationSize - 1 );
	for ( i = 0; i < populationSize / 2; i++ ) {
		parent1 = validList[permutation[2 * i]];
		parent2 = validList[permutation[2 * i + 1]];
		child1 = invalidList[2 * i];
		child2 = invalidList[2 * i + 1];
		weight1 = 0;
		weight2 = 0;
		cost1 = 0;
		cost2 = 0;

		polymorphismModify(solution[parent1].Chromosome, solution[parent2].Chromosome);

		for ( j = 0; j < length; j++ ) {
			if ( randGenerator.uniformInt( 0, 1 ) ) {
				solution[child1].Chromosome[j] = solution[parent1].Chromosome[j];
				solution[child2].Chromosome[j] = solution[parent2].Chromosome[j];
				if ( solution[parent1].Chromosome[j] == 1 ) {
					weight1 = weight1 + 1 * j;
					cost1++;
				}
				if ( solution[parent2].Chromosome[j] == 1 ) {
					weight2 = weight2 + 1 * j;
					cost2++;
				}
			} else {
				solution[child1].Chromosome[j] = solution[parent2].Chromosome[j];
				solution[child2].Chromosome[j] = solution[parent1].Chromosome[j];
				if ( solution[parent2].Chromosome[j] == 1 ) {
					weight1 = weight1 + 1 * j;
					cost1++;
				}
				if ( solution[parent1].Chromosome[j] == 1 ) {
					weight2 = weight2 + 1 * j;
					cost2++;
				}
			}
		}
		solution[child1].y = cost1;
		solution[child2].y = cost2;
	}

	// Parallelly call DNN to get MSE
	#pragma omp parallel for
	for (i = 0; i < populationSize; i++) {
		int tid = omp_get_thread_num();
		solution[invalidList[i]].x = getMSE(invalidList[i], tid);
	}
}

void noneDominateSort() {
	int check = 0, currentPart = 1, totalCount = 0;
	bool isOver = false;
	int i, j;

	// overshooting nondominate sort
	validCount = 0;
	while ( !isOver ) {
		for ( i = 0; i < populationSize * 2; i++ ) {
			if ( solution[i].isChoose == true ) {
				if ( solution[i].part == (currentPart - 1) ) {
					validList[validCount++] = i;
					solution[i].age++;
				}
				continue;
			}
			check = 0;
			for ( j = 0; j < populationSize * 2; j++ ) {
				if ( (solution[j].isChoose == true && solution[j].part < currentPart) || i == j ) {
					continue;
				}
				check = noneDominate( solution[i], solution[j] );
				if ( check == 1 ) {
					break;
				}
			}
			if ( check == 0 ) {
				solution[i].part = currentPart;
				solution[i].isChoose = true;
				totalCount++;
			}
		}
		if ( totalCount > populationSize ) {
			isOver = true;
			edgePart = currentPart;
		} else {
			currentPart++;
		}
	}

	// prepare indexPart for crowding distance sort
	indexPart.clear();
	for ( i = 0; i < populationSize * 2; i++ ) {
		if ( solution[i].isChoose == true && solution[i].part == edgePart ) {
			indexPart.push_back( i );
		}
	}
}

int noneDominate( const sol &a, const sol &b ) {
	if ( (b.x >= a.x && b.y < a.y) || (b.x > a.x && b.y <= a.y) ) {
		return 1;
	}
	return 0;
}

int lessIndirectCrowdingDis( const void *a, const void *b )
{
	double first = crowdingDisPart[*(int *) a];
	double second = crowdingDisPart[*(int *) b];
	if ( first < second ) {
		return -1;
	}
	else if (first == second) {
		return 0;
	}
	else {
		return 1;
	}
}

void crowdingDisSort() {
	int i, j;
	//static double pairDistance[populationSize * 2][populationSize * 2] = {0};
	double *pairDistance;
	int iTempA, iTempB;
	double tempA = DBL_MAX, tempB = DBL_MAX;
	double minDis = DBL_MAX, secondMinDis = DBL_MAX, tempDis;

	pairDistance = (double *) malloc( indexPart.size() * indexPart.size() * sizeof( double ) );
	for ( i = 0; i < indexPart.size() - 1; i++ ) {
		for ( j = i + 1; j < indexPart.size(); j++ ) {
			pairDistance[i * indexPart.size() + j] = (solution[indexPart[i]].x - solution[indexPart[j]].x)*(solution[indexPart[i]].x - solution[indexPart[j]].x) + (solution[indexPart[i]].y - solution[indexPart[j]].y)*(solution[indexPart[i]].y - solution[indexPart[j]].y);
			pairDistance[j * indexPart.size() + i] = pairDistance[i * indexPart.size() + j];
		}
	}

	for ( i = 0; i < indexPart.size(); i++ ) {
		iTempA = -1;
		minDis = DBL_MAX;
		secondMinDis = DBL_MAX;
		for ( j = 0; j < indexPart.size(); j++ ) {
			if ( i == j )
				continue;
			tempDis = pairDistance[i * indexPart.size() + j];
			if ( tempDis < minDis ) {
				if ( iTempA == -1 ) {
					secondMinDis = DBL_MAX;
					minDis = tempDis;
					iTempA = j;
				} else {
					iTempB = iTempA;
					secondMinDis = minDis;
					minDis = tempDis;
					iTempA = j;
				}
			} else if ( tempDis < secondMinDis ) {
				secondMinDis = tempDis;
				iTempB = j;
			}
		}
		crowdingDisPart[indexPart[i]] = pairDistance[i * indexPart.size() + j];
	}
	qsort( indexPart.data(), indexPart.size(), sizeof( int ), lessIndirectCrowdingDis );
	free( pairDistance );
}

bool converge() {
	static int numFirstLast = 0;
	int numFirst = 0;
	int i;

	for ( i = 0; i < populationSize; i++ ) {
		if ( solution[i].part != 1 ) {
			break;
		}
	}
	printf( "numF = %d\n", i );

	return 1;
}

double getMSE( int indexChromosome, int tid ) {
	double MSE = 0;
	FILE *fp = fopen( (string("request_") + to_string(tid)).c_str(), "w" );
	int i = 0, j = 0, k = 0;

	for ( i = 0; i < length; i++ ) {
		if ( solution[indexChromosome].Chromosome[i] == 1 )
			j++;
	}
	//printf( "%d\n", j );
	fprintf( fp, "%d\n", j );
	for ( i = 0; i < inputNode; i++ ) {
		for ( j = 0; j < outputNode; j++ ) {
			if ( solution[indexChromosome].Chromosome[k] == 1 ) {
				//printf( "%d %d\n", j, i );
				fprintf( fp, "%d %d\n", j, i );
			}
			k++;
		}
	}
	fclose( fp );
	int status = system( (string("python python/sparse_autoencoder.pyc ") + to_string(tid)).c_str() );
	if (status != 0)
		exit(status);
	fp = fopen( (string("reply_") + to_string(tid)).c_str(), "r" );
	fscanf( fp, "%lf", &MSE );
	fclose( fp );

	return -MSE;
}

void storeBest( int indexChromosome, int indexFile ) {
	FILE *fp1 = fopen( "request_0", "w" ), *fp2;
	char filename[40];
	char filename2[40];
	int i = 0, j = 0, k = 0;

	sprintf( filename, "data/topology_%s_g%d-%d", jobname, curentGeneration, indexFile );
	sprintf( filename2, "data/topology_%s-%d", jobname, indexFile );
	fp2 = fopen( filename, "w" );
	for ( i = 0; i < length; i++ ) {
		if ( solution[indexChromosome].Chromosome[i]) {
			j++;
		}
	}
	//printf( "%d\n", j );
	fprintf( fp2, "%lf\n%d\n", -solution[indexChromosome].x, solution[indexChromosome].y );
	fprintf( fp1, "%d\n", j );
	fprintf( fp2, "%d\n", j );
	for ( i = 0; i < inputNode; i++ ) {
		for ( j = 0; j < outputNode; j++ ) {
			if ( solution[indexChromosome].Chromosome[k] == 1 ) {
				//printf( "%d %d\n", j, i );
				fprintf( fp1, "%d %d\n", j, i );
				fprintf( fp2, "%d %d\n", j, i );
			}
			k++;
		}
	}
	//printf( "output_I%dO%d-%d", inputNode, outputNode, indexFile );
	fprintf( fp1, "data/output_%s-%d", jobname, indexFile );
	fclose( fp1 );
	fclose( fp2 );
	system((string("cp ") + filename + " " + filename2).c_str());
	system( "python python/sparse_autoencoder.pyc 0" );
}
