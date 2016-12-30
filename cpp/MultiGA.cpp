#include<stdio.h>
#include<iostream>
#include<math.h>
#include <vector>
#include<time.h>
#include <algorithm> 
using namespace std;
struct sol{
	int x;//accuracy
	int y;//cost
	int part;
	int crowdingDist;
	bool isChoose=false;
	vector<int>Chromosome;

};

void initial();
void selection();
int noneDominate(sol a,sol b);
void noneDominateSort();
void crowdingDisSort();
bool checkOver();
int cmp(const void *a, const void *b);
int generation =5;
int populationSize = 10;
int edgePart;
int length = 4;
bool struct_cmp_by_crowdingDist(const sol &a, const sol &b);
bool  struct_cmp_by_part(const sol &a, const sol &b);
void crossover();
vector<sol> solution;
vector<sol>thePart;
vector<sol>nextGeneration;
int main(){
	int i;
	initial();
	/*for (i = 0; i < 10; i++)
	{
		printf("solution[%d]:%d,%d\n", i, solution[i].x, solution[i].y);
	}*/
	i = 0;

	while (i <generation){
		crossover();
		selection();
		i++;
		thePart.clear();
	}
	scanf_s("%d", &i);
	return 0;
}
void initial(){
	solution.resize(populationSize * 2);
	int i=0,j,tp,weight=0,cost=0;
	srand((unsigned)time(NULL));
	
	for (i = 0; i <populationSize; i++){
		for (j = 0; j < length; j++){
			tp = rand() % 10 + 1;
			if (tp > 5)
			{
				solution[i].Chromosome.push_back(1);
				weight=weight+1*j;
				cost++;
			}
			else{
				solution[i].Chromosome.push_back(0);
			}
			
		}
		solution[i].x = weight;
		solution[i].y = cost;
		weight = 0;
		cost = 0;
	}
}

void selection(){
	int i,currentNumber=0,j;
	//vector<sol>afterSortingPart;
	noneDominateSort();
	for (i = 0; i < populationSize; i++){
		if (solution[i].part < edgePart)
		{
			nextGeneration.push_back(solution[i]);
			currentNumber++;
		}
		else
			break;
	}
	if (thePart.size()>2)
	 crowdingDisSort();
	 for (i = 0; i < thePart.size(); i++){
		 if (currentNumber<populationSize)
		 {
			 nextGeneration.push_back(thePart.at(i));
			 currentNumber++;
		 }
		 else
			 break;
	 }
	 solution.clear();
	// solution.resize(20);
	 
	 for (i = 0; i < nextGeneration.size(); i++){
		 printf("next generation[%d]:%d,%d\n", i, nextGeneration.at(i).x, nextGeneration.at(i).y );
		 printf("next generation[%d] chromosome:",i);
		 for ( j = 0; j < length; j++)
		 {
			 printf("%d", nextGeneration[i].Chromosome[j]);
		 }
		 printf("\n");
	 }
	 printf("------------------------------------------------\n");
	 solution.assign(nextGeneration.begin(), nextGeneration.end());
	// nextGeneration.assign(solution.begin(), solution.begin()+10);
	/* for (i = 0; i < nextGeneration.size(); i++){
		 printf("next generation[%d]:%d,%d\n", i, solution.at(i).x, solution.at(i).y);
		 printf("next generation[%d] chromosome:", i);
		 for (j = 0; j < length; j++)
		 {
			 printf("%d", solution.at(i).Chromosome.at(j));
		 }
		 printf("\n");

	 }*/
	 solution.resize(populationSize*2);
	 nextGeneration.clear();
	 thePart.clear();
	
}
void crossover(){
	int i, j,parent1,parent2,tp,weight=0,cost=0;
	srand((unsigned)time(NULL));
	for (i = populationSize; i < populationSize * 2; i++)
	{
		parent1 = rand() % populationSize;
		parent2 = rand() % populationSize;
		for (j = 0; j < length; j++)
		{
			tp = rand() % 10;
			if (tp > 5)
			{
				solution.at(i).Chromosome.push_back(solution.at(parent1).Chromosome.at(j));
				if (solution[parent1].Chromosome[j] == 1)
				{
					weight = weight + 1 * j;
					cost++;
					
				}
			}
			else
			{
				solution.at(i).Chromosome.push_back(solution.at(parent2).Chromosome.at(j));
				if (solution[parent2].Chromosome[j] == 1)
				{
					weight = weight + 1 * j;
					cost++;

				}
			}
		}
		solution.at(i).x = weight;
		solution.at(i).y = cost;
		weight = 0;
		cost = 0;
	}
	
}


void noneDominateSort(){
	int i, j, check = 0, part = 1, totalCount = 0, currentPartCount=0;
	bool isBreak = false;
	bool isOver = false;
	bool isRecord = false;
	while (isOver == false)
	{
		for (i = 0; i < populationSize*2; i++){
			if (solution.at(i).isChoose == true)
				continue;
			for (j = 0; j < populationSize*2; j++)
			{
				if (solution.at(j).isChoose == true)
					continue;
				if (i == j)
					continue;
				if (check == 1)
				{
					isBreak = true;
					break;
				}
				check = noneDominate(solution.at(i),solution.at(j));
			}
			if (isBreak == false)
			{
				solution.at(i).part = part;
				currentPartCount++;
			}
			check = 0;
			isBreak= false;
		}
		for (i = 0; i <populationSize*2; i++){
			if (solution.at(i).part == part)
				solution.at(i).isChoose = true;
		}
		
		totalCount = totalCount + currentPartCount;
		if (totalCount>populationSize &&isRecord == false)
		{
			edgePart = part;
			isRecord = true;
		}
			part++;
		currentPartCount = 0;
		isOver = checkOver();
	}
	/*for (int i = 0; i<populationSize*2; i++){
		printf("solution[%d]:%d,%d,%d\n", i, solution[i].x, solution[i].y,solution[i].part);
	}*/
	sort(solution.begin(), solution.end(), struct_cmp_by_part);
	//qsort(solution, populationSize, sizeof(solution[0]), cmp);
	for (i = 0; i < populationSize*2; i++)
	{
		if (solution[i].part == edgePart)
			thePart.push_back(solution[i]);
		//printf("solution After sorting[%d]:%d,%d,%d\n", i, solution[i].x, solution[i].y,solution[i].part);
	}
	/*for (int i = 0; i<thePart.size(); i++){
		printf("certain solution[%d]:%d,%d,%d\n", i, thePart[i].x, thePart[i].y, thePart[i].part);
	}
		*/

}
int noneDominate(sol a, sol b){
	if (b.x>a.x && b.y < a.y)
	{
		return 1;
	}
	return 0;
}
bool checkOver(){
	int i;
	bool isOver=true;
	for (i = 0; i <populationSize*2; i++)
	{
		if (solution.at(i).isChoose == false)
		{
			isOver = false;
			return isOver;
		}
	}
	return isOver;

}
void crowdingDisSort(){
	int i,j;
	sol tempA, tempB;
	int minDis=999, secondMinDis = 999;
	int tempDis=0;
	
	for (i = 0; i < thePart.size(); i++){
		tempA.x = 999;
		for (j = 0; j < thePart.size(); j++){
			if (i == j)
				continue;
			tempDis = (thePart.at(i).x - thePart.at(j).x)*(thePart.at(i).x - thePart.at(j).x) + (thePart.at(i).y - thePart.at(j).y)*(thePart.at(i).y - thePart.at(j).y);
					
			if (tempDis < minDis){
				if (tempA.x == 999)
				{
					
					secondMinDis = 999;
					minDis = tempDis;
					tempA = thePart.at(j);
				}
				else{
					tempB = tempA;
					secondMinDis = minDis;
					minDis = tempDis;
					tempA = thePart.at(j);
				}
				
			}
			else if (tempDis < secondMinDis){
				secondMinDis = tempDis;
				tempB = thePart.at(j);
			}
		}
		
		minDis = 999;
		secondMinDis = 999;
		thePart.at(i).crowdingDist = (tempA.x - tempB.x)*(tempA.x - tempB.x) + (tempA.y - tempB.y)*(tempA.y - tempB.y);
		
	}
	
	sort(thePart.begin(), thePart.begin() + thePart.size(), struct_cmp_by_crowdingDist);
	/*for (int i = 0; i<thePart.size(); i++){
		printf("certain solution After sorting[%d]:%d,%d,%d\n", i, thePart[i].x, thePart[i].y,thePart[i].crowdingDist);
	}*/
	
}
int cmp(const void *a, const void *b)
{
	return ((sol *)a)->part - ((sol *)b)->part;
}
bool struct_cmp_by_crowdingDist(const sol &a, const sol &b)
{
	return a.crowdingDist> b.crowdingDist;
}
bool struct_cmp_by_part(const sol &a, const sol &b)
{
	return a.part< b.part;
}

