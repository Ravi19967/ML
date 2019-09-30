#include<iostream>
#include <cmath>
using namespace std;

int main(){
	float pi = 0;
  float errorFactor = 0.0001
	int epochs = 0;
	int pointsCircle = 0;
	int pointsSquare = 0;
	cin>>epochs;
	for(int i=0 ; i<epochs ; i++){
		float x = static_cast <float> (rand()) / static_cast <float> (RAND_MAX);
		float y = static_cast <float> (rand()) / static_cast <float> (RAND_MAX);
		if(x*x + y*y <= 1)
		    pointsCircle++;
		pointsSquare++;
		pi = (float(4)*pointsCircle)/(pointsSquare);
		if(i%1000 == 0){
		    if((abs((pi*0.5*0.5)-0.78539816339)) < errorFactor){
		        cout<<pi<<endl;
		        break;
		    }
		}
	}
}
