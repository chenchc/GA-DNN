#include<stdio.h>
#include<iostream>
#include<math.h>
#include <vector>
#include<time.h>
#include <algorithm>
#include <cfloat>
#include "myrand.h"
#include <functional>

using namespace std;

struct sol {
	double x;//accuracy
	int y;//cost
	int part;
	double crowdingDist;
	bool isChoose = false;
	vector<bool>Chromosome;
};

void initial();
void selection();
int noneDominate( const sol &, const sol & );
void noneDominateSort();
void crowdingDisSort();
void crossover();
bool converge();
double getMSE( int );

const int generation = 10;
const int populationSize = 10;
const int inputNode = 49;
const int outputNode = 40;
const int length = inputNode * outputNode;

int edgePart;
MyRand randGenerator;

vector<sol> solution( populationSize * 2 );
double crowdingDisPart[populationSize * 2] = {0};
vector<int> indexPart;
int validList[populationSize] = {0};
int invalidList[populationSize] = {0};
int validCount;
int main() {
	int i, j, k;
	char name[20] = "mnist.csv";
	FILE *fp = fopen( "spec", "w" );

	printf( "%s\n%d\n%d\n", name, inputNode, outputNode );
	fprintf( fp, "%s\n%d\n%d\n", name, inputNode, outputNode );
	fclose( fp );

	initial();

	i = 0;
	while ( i < generation ) {
		crossover();
		//getMSE();
		selection();
		i++;
		/*for ( j = 0; j < populationSize; j++ ) {
			printf( "%d ", solution[j].part );
		}
		printf( "\n" );*/
		k = 0;
		for ( j = 0; j < populationSize; j++ ) {
			printf( "%d-%lf,%d ", solution[validList[j]].part, solution[validList[j]].x, solution[validList[j]].y );
			if ( solution[validList[j]].part == 1 ) {
				k++;
			}
		}
		printf( "\ngenration %d\nnum of first = %d\n\n", i, k );
		//converge();
	}

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
}
void initial() {
	int i = 0, j, weight = 0, cost = 0;

	for ( i = 0; i <populationSize; i++ ) {
		validList[i] = i;
		invalidList[i] = i + populationSize;
		for ( j = 0; j < length; j++ ) {
			solution[i + populationSize].Chromosome.push_back( 0 );
			if ( randGenerator.uniformInt( 0, 1 ) ) {
				solution[i].Chromosome.push_back( 1 );
				cost++;
			} else {
				solution[i].Chromosome.push_back( 0 );
			}

		}
		solution[i].x = getMSE(i);
		solution[i].y = cost;
		cost = 0;
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
		} else
			break;
	}
	for ( i = 0; i < populationSize; i++ ) {
		invalidTable[validList[i]] = 1;
	}
	j = 0;
	for ( i = 0; i < 2 * populationSize; i++ ) {
		if ( !invalidTable[i] ) {
			invalidList[j++] = i;
		}
		// initialization for next generation
		solution[i].isChoose = false;
	}
	indexPart.clear();
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
	static int permutation[populationSize] = {0};

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

		//polymorphismModify(solution[parent1].Chromosome, solution[parent2].Chromosome);

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
		//solution[child1].x = weight1;
		solution[child1].x = getMSE( child1 );
		solution[child1].y = cost1;
		//solution[child2].x = weight2;
		solution[child2].x = getMSE( child2 );
		solution[child2].y = cost2;
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

	// prepare indexPart for crowdind distance sort
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
	double tempA = 999, tempB = 999;
	double minDis = 999, secondMinDis = 999, tempDis = 0;

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
		crowdingDisPart[i] = pairDistance[i * indexPart.size() + j];
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

double getMSE( int indexChromosome ) {
	double MSE = 0;
	FILE *fp = fopen( "request", "w" );
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
	int status = system( "python python/sparse_autoencoder.pyc" );
	fp = fopen( "reply", "r" );
	fscanf( fp, "%lf", &MSE );
	fclose( fp );

	return -MSE;
}
