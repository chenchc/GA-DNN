#include <stdio.h>
#include <string.h>
#include <vector>
#include <limits.h>
#include <cstdlib>
#include <cfloat>

typedef struct {
	double performance;
	int cost;
	int part;
	bool isChoose = false;
} RESULT;

int noneDominate( const RESULT &, const RESULT & );
void noneDominateSort( void );
void crowdingDisSort( void );
int lessIndirectCrowdingDis( const void *, const void * );

const int nElder = 3;

RESULT mother[nElder];
RESULT child[nElder * nElder];
RESULT winner[nElder];
int indexWinner[nElder];
double crowdingDisPart[nElder * nElder] = {0};
std::vector<int> indexPart;
int edgePart;
int main( int argc, char *argv[]) {   // argv[1] is mother, argv[2] is child
	char filename[50];
	char motherFilename[100];
	FILE *fp;
	int i, j;

	strcpy( motherFilename, "data/topology_" );
	strcat( motherFilename, argv[1] );
	for ( i = 0; i < nElder; i++ ) {
		sprintf( filename, "%s-%d", motherFilename, i + 1 );
		fp = fopen( filename, "r" );
		fscanf( fp, "%lf", &mother[i].performance );
		fscanf( fp, "%d", &mother[i].cost );
		fclose( fp );
	}
	strcpy( motherFilename, "data/topology_" );
	strcat( motherFilename, argv[2] );
	strcat( motherFilename, "J" );
	for ( i = 0; i < nElder; i++ ) {
		for ( j = 0; j < nElder; j++ ) {
			sprintf( filename, "%s%d-%d", motherFilename, i + 1, j + 1 );
			fp = fopen( filename, "r" );
			fscanf( fp, "%lf", &child[i * nElder + j].performance );
			fscanf( fp, "%d", &child[i * nElder + j].cost );
			fclose( fp );
			child[i * nElder + j].performance = child[i * nElder + j].performance + mother[i].performance;
			child[i * nElder + j].cost = child[i * nElder + j].cost + mother[i].cost;
			child[i * nElder + j].part = INT_MAX;
			child[i * nElder + j].isChoose = false;
		}
	}

	noneDominateSort();
	j = 0;
	for ( i = 0; i < nElder * nElder; i++ ) {
		if ( child[i].part < edgePart ) {
			winner[j] = child[i];
			indexWinner[j] = i;
			j++;
		}
	}
	crowdingDisSort();
	for ( i = 0; i < indexPart.size(); i++ ) {
		if ( j < nElder ) {
			winner[j] = child[indexPart[i]];
			indexWinner[j] = indexPart[i];
			j++;
		} else
			break;
	}

	for ( i = 0; i < nElder; i++ ) {
		strcpy( motherFilename, "cp data/output_" );
		strcat( motherFilename, argv[2] );
		sprintf( filename, "J%d-%d", indexWinner[i] / nElder + 1, indexWinner[i] % nElder + 1 );
		strcat( motherFilename, filename );
		strcat( motherFilename, " data/output_" );
		strcat( motherFilename, argv[2] );
		sprintf( filename, "-%d", i + 1 );
		strcat( motherFilename, filename );
		//printf( "%s\n", motherFilename );
		system( motherFilename );
	}

	for ( i = 0; i < nElder; i++ ) {
		strcpy( motherFilename, "cp data/topology_" );
		strcat( motherFilename, argv[2] );
		sprintf( filename, "J%d-%d", indexWinner[i] / nElder + 1, indexWinner[i] % nElder + 1 );
		strcat( motherFilename, filename );
		strcat( motherFilename, " data/topology_" );
		strcat( motherFilename, argv[2] );
		sprintf( filename, "-%d", i + 1 );
		strcat( motherFilename, filename );
		//printf( "%s\n", motherFilename );
		system( motherFilename );
	}

	strcpy( filename, "data/shrink_" );
	strcat( filename, argv[2] );
	fp = fopen( filename, "w" );
	for ( i = 0; i < nElder; i++ ) {
		strcpy( motherFilename, argv[2] );
		sprintf( filename, "J%d-%d", indexWinner[i] / nElder + 1, indexWinner[i] % nElder + 1 );
		strcat( motherFilename, filename );
		//printf( "%s\n", motherFilename );
		fprintf( fp, "%s\n", motherFilename );
	}
	fclose( fp );
}

void noneDominateSort( void ) {
	int check = 0, currentPart = 1, totalCount = 0;
	bool isOver = false;
	int i, j;

	// overshooting nondominate sort
	while ( !isOver ) {
		for ( i = 0; i < nElder * nElder; i++ ) {
			if ( child[i].isChoose == true ) {
				continue;
			}
			check = 0;
			for ( j = 0; j < nElder * nElder; j++ ) {
				if ( (child[j].isChoose == true && child[j].part < currentPart) || i == j ) {
					continue;
				}
				check = noneDominate( child[i], child[j] );
				if ( check == 1 ) {
					break;
				}
			}
			if ( check == 0 ) {
				child[i].part = currentPart;
				child[i].isChoose = true;
				totalCount++;
			}
		}
		if ( totalCount > nElder ) {
			isOver = true;
			edgePart = currentPart;
		} else {
			currentPart++;
		}
	}

	// prepare indexPart for crowding distance sort
	indexPart.clear();
	for ( i = 0; i < nElder * nElder; i++ ) {
		if ( child[i].isChoose == true && child[i].part == edgePart ) {
			indexPart.push_back( i );
		}
	}
}

int noneDominate( const RESULT &a, const RESULT &b ) {
	if ( (b.performance >= a.performance && b.cost < a.cost) || (b.performance > a.performance && b.cost <= a.cost) ) {
		return 1;
	}
	return 0;
}

int lessIndirectCrowdingDis( const void *a, const void *b ) {
	double first = crowdingDisPart[*(int *) a];
	double second = crowdingDisPart[*(int *) b];

	if ( first < second ) {
		return 1;
	} else if ( first == second ) {
		return 0;
	} else {
		return -1;
	}
}

void crowdingDisSort( void ) {
	int i, j;
	double *pairDistance;
	int iTempA, iTempB;
	double tempA = DBL_MAX, tempB = DBL_MAX;
	double minDis = DBL_MAX, secondMinDis = DBL_MAX, tempDis;

	pairDistance = (double *) malloc( indexPart.size() * indexPart.size() * sizeof( double ) );
	for ( i = 0; i < indexPart.size() - 1; i++ ) {
		for ( j = i + 1; j < indexPart.size(); j++ ) {
			pairDistance[i * indexPart.size() + j] = (child[indexPart[i]].performance - child[indexPart[j]].performance)*(child[indexPart[i]].performance - child[indexPart[j]].performance) + (child[indexPart[i]].cost - child[indexPart[j]].cost)*(child[indexPart[i]].cost - child[indexPart[j]].cost);
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
