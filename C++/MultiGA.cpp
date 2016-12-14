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
	bool isChoose;
};

void initial();
void selection();
int noneDominate(sol a,sol b);
void noneDominateSort();
void crowdingDisSort();
bool checkOver();
int cmp(const void *a, const void *b);
int populationSize = 10;
int edgePart;
bool struct_cmp_by_sol(const sol &a, const sol &b);
sol solution[10];
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
	while (i < 100){
		selection();
		
	}
	return 0;
}
void initial(){
	int i;
	srand((unsigned)time(NULL));
	for (i = 0; i <populationSize; i++){
		solution[i].x = rand() % 30 + 1;
		solution[i].y = rand() % 30 + 1;
		}
}

void selection(){
	int i,currentNumber=0;
	vector<sol>afterSortingPart;
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
		 if (currentNumber<populationSize/2)
		 {
			 nextGeneration.push_back(thePart.at(i));
			 currentNumber++;
		 }
		 else
			 break;
	 }
	 for (i = 0; i < nextGeneration.size(); i++){
		 printf("next generation[%d]:%d,%d\n", i, nextGeneration.at(i).x, nextGeneration.at(i).y );
	 }

	 scanf_s("%d", &i);
}


void noneDominateSort(){
	int i, j, check = 0, part = 1, partCount = 0, currentPartCount=0;
	bool isBreak = false;
	bool isOver = false;
	bool isRecord = false;
	while (isOver == false)
	{
		for (i = 0; i < populationSize; i++){
			if (solution[i].isChoose == true)
				continue;
			for (j = 0; j < populationSize; j++)
			{
				if (solution[j].isChoose == true)
					continue;
				if (i == j)
					continue;
				if (check == 1)
				{
					isBreak = true;
					break;
				}
				check = noneDominate(solution[i], solution[j]);
			}
			if (isBreak == false)
			{
				solution[i].part = part;
				currentPartCount++;
			}
			check = 0;
			isBreak= false;
		}
		for (i = 0; i <populationSize; i++){
			if (solution[i].part == part)
				solution[i].isChoose = true;
		}
		
		partCount = partCount + currentPartCount;
		if (partCount>populationSize / 2&&isRecord==false)
		{
			edgePart = part;
			isRecord = true;
		}
			part++;
		currentPartCount = 0;
		isOver = checkOver();
	}
	qsort(solution, populationSize, sizeof(solution[0]), cmp);
	for (i = 0; i < populationSize; i++)
	{
		if (solution[i].part == edgePart)
			thePart.push_back(solution[i]);
		printf("solution[%d]:%d,%d,%d\n", i, solution[i].x, solution[i].y,solution[i].part);
	}
	for (int i = 0; i<thePart.size(); i++){
		printf("certain solution[%d]:%d,%d,%d\n", i, thePart[i].x, thePart[i].y, thePart[i].part);
	}
		

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
	for (i = 0; i <populationSize; i++)
	{
		if (solution[i].isChoose == false)
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
	
	sort(thePart.begin(), thePart.begin() + thePart.size(), struct_cmp_by_sol);
	for (int i = 0; i<thePart.size(); i++){
		printf("certain solution After sorting[%d]:%d,%d,\n", i, thePart[i].x, thePart[i].y);
	}
	
}
int cmp(const void *a, const void *b)
{
	return ((sol *)a)->part - ((sol *)b)->part;
}
bool struct_cmp_by_sol(const sol &a, const sol &b)
{
	return a.crowdingDist< b.crowdingDist;
}
