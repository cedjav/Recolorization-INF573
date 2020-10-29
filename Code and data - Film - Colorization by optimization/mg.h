#include "fmg.h"
#include "tensor3d.h"
#include <fstream>
#include <vector>
#include <algorithm>
#ifndef VSMKGRD_HH
#define VSMKGRD_HH

using namespace std;

class MG : public TensorField{
public:
	// Constructors
	MG() {_nebl=NULL; }
  
	MG(int nx, int ny,int nz, int depth) {
		_nebl=NULL;
		set(nx, ny,nz, depth); // See below
	}

	// Destructor
	~MG() {
		if (_nebl)
			delete [] _nebl;
	}
	
	void set(int nx, int ny,int nz,int depth) {
		_deg=1;
		// Initialise sizes on x, y, z and depth
		_depth=depth;
		_len_x = nx ;
		_len_y = ny ;
		_len_z = nz;
		
		int i,j;
		
		if (_nebl)
			delete [] _nebl;
		
		_nebl=new Tensore3dVP[ _depth ];
		_P.resize(_depth);
		_Div.resize(_depth);
		_I.resize(_depth);
		_G.resize(_depth);
		
		for (i=0; i<_depth; i++)
			_G[i].resize(_deg+1);
		
		for(i = 0 ; i < _depth ; i++) { // Creation of different layers each time 50% of the size of the previous one. To compute optical flow? (P I et Div are vectors of Tensor3d)
			_P[i].set((int)(_len_x * pow(2,-i)), (int)(_len_y * pow(2,-i)), _len_z) ;
			_P[i].clear(0) ;
			_I[i].set((int)(_len_x * pow(2,-i)), (int)(_len_y * pow(2,-i)),_len_z) ;
			_I[i].clear(0) ;
			_Div[i].set((int)(_len_x * pow(2,-i)), (int)(_len_y * pow(2,-i)), _len_z) ;
			_Div[i].clear(0) ;
      
     
			for ( j=0; j<=_deg; j++){ // Idem for G but all all the degrees G is a vector or vector of Tensor 3d
				_G[i][j].set((int)(_len_x * pow(2,-i)), (int)(_len_y * pow(2,-i)),_len_z) ;
				_G[i][j].clear(0);
			}
		}

		_perms_num=1;   
		_perms.resize(_perms_num); 
		_perms[0].resize(27);  
		_perms_len=27;	//perms is an array of size 1 * 27 * 3 made to handle all the points in a (-1;0;+1)**3 around one point (?)
		for (i=0 ; i< _perms_len; i++) 
			_perms[0][i].resize(0);

		_perms[0][0].push_back(0);_perms[0][0].push_back(0);_perms[0][0].push_back(0);
		_perms[0][1].push_back(1);_perms[0][1].push_back(0);_perms[0][1].push_back(0);
		_perms[0][2].push_back(0);_perms[0][2].push_back(1);_perms[0][2].push_back(0);
		_perms[0][3].push_back(1);_perms[0][3].push_back(1);_perms[0][3].push_back(0);
		_perms[0][4].push_back(0);_perms[0][4].push_back(-1);_perms[0][4].push_back(0);
		_perms[0][5].push_back(-1);_perms[0][5].push_back(0);_perms[0][5].push_back(0);
		_perms[0][6].push_back(-1);_perms[0][6].push_back(-1);_perms[0][6].push_back(0);
		_perms[0][7].push_back(1);_perms[0][7].push_back(-1);_perms[0][7].push_back(0);
		_perms[0][8].push_back(-1);_perms[0][8].push_back(1);_perms[0][8].push_back(0);
		_perms[0][9].push_back(0);_perms[0][9].push_back(0);_perms[0][9].push_back(1);
		_perms[0][10].push_back(1);_perms[0][10].push_back(0);_perms[0][10].push_back(1);
		_perms[0][11].push_back(0);_perms[0][11].push_back(1);_perms[0][11].push_back(1);
		_perms[0][12].push_back(1);_perms[0][12].push_back(1);_perms[0][12].push_back(1);
		_perms[0][13].push_back(0);_perms[0][13].push_back(-1);_perms[0][13].push_back(1);
		_perms[0][14].push_back(-1);_perms[0][14].push_back(0);_perms[0][14].push_back(1);
		_perms[0][15].push_back(-1);_perms[0][15].push_back(-1);_perms[0][15].push_back(1);
		_perms[0][16].push_back(1);_perms[0][16].push_back(-1);_perms[0][16].push_back(1);
		_perms[0][17].push_back(-1);_perms[0][17].push_back(1);_perms[0][17].push_back(1);
		_perms[0][18].push_back(0);_perms[0][18].push_back(0);_perms[0][18].push_back(-1);
		_perms[0][19].push_back(1);_perms[0][19].push_back(0);_perms[0][19].push_back(-1);
		_perms[0][20].push_back(0);_perms[0][20].push_back(1);_perms[0][20].push_back(-1);
		_perms[0][21].push_back(1);_perms[0][21].push_back(1);_perms[0][21].push_back(-1);
		_perms[0][22].push_back(0);_perms[0][22].push_back(-1);_perms[0][22].push_back(-1);
		_perms[0][23].push_back(-1);_perms[0][23].push_back(0);_perms[0][23].push_back(-1);
		_perms[0][24].push_back(-1);_perms[0][24].push_back(-1);_perms[0][24].push_back(-1);
		_perms[0][25].push_back(1);_perms[0][25].push_back(-1);_perms[0][25].push_back(-1);
		_perms[0][26].push_back(-1);_perms[0][26].push_back(1);_perms[0][26].push_back(-1);
		// Reminder : rewrite that in a nicer way later
		/* Finaly, _perms contains the following :
			0  ->  0   0   0
			1  ->  1   0   0
			2  ->  0   1   0
			3  ->  1   1   0
			4  ->  0  -1   0
			5  -> -1   0   0
			6  -> -1  -1   0
			7  ->  1  -1   0
			8  -> -1   1   0
			9  ->  0   0   1
			10 ->  1   0   1
			11 ->  0   1   1
			12 ->  1   1   1
			13 ->  0  -1   1
			14 -> -1   0   1
			15 -> -1  -1   1
			16 ->  1  -1   1
			17 -> -1   1   1
			18 ->  0   0  -1
			19 ->  1   0  -1
			20 ->  0   1  -1
			21 ->  1   1  -1
			22 ->  0  -1  -1
			23 -> -1   0  -1
			24 -> -1  -1  -1
			25 ->  1  -1  -1
			26 -> -1   1  -1
			All the values (x,y,z) are taken once and only once in (-1 0 +1)
		*/
	}
  
	Tensor3d& Div(int level=0) { return _Div[level]; } ;
	void setDepth(int d) { _depth = d; cout << "Depth :" << d << endl; }; //For begugginf purpose
  
	void setI(Tensor3d& II) {
		_I[0] = II ; // At depth=0, it's just II

		int nx = _len_x ;
		int ny = _len_y;
		int nz = _len_z;
		
		for(int d = 1 ; d < _depth ; d++) {
			// On next lawyers : set each point to 0 or 1 (1 iif all the 4 points in laywer d-1 are already 1)
			nx /= 2 ;
			ny /= 2 ;
			for(int z = 0 ; z < nz ; z++)
				for(int y = 0 ; y < ny ; y++)
					for(int x = 0 ; x < nx ; x++) {
						if(_I[d-1](2*x,2*y,z) +  _I[d-1](2*x+1,2*y,z) + _I[d-1](2*x,2*y+1,z) + _I[d-1](2*x+1,2*y+1,z) < 4)
							_I[d](x,y,z) = 0 ;
						else
							_I[d](x,y,z) = 1 ;
						}
		}
	}

	void setG(Tensor3d& GG) {
		_G[0][1] = GG ;		// GG on depth 0 and [1]
		_G[0][0].clear(1); // All at 1 on depth 0 but [0]

		int nx = _len_x ;
		int ny = _len_y ;
		int nz = _len_z;
    
		for(int lvl = 1 ; lvl < _depth ; lvl++) {
			_G[lvl][0].clear(1); // As for depth 0
			restrict(_G[lvl-1][1],_G[lvl][1]); // Define in mg.cpp
		}
    
	}

	void setFlow(Tensor3d& dx,Tensor3d& dy,Tensor3d& idx,Tensor3d& idy);
  
	void smooth(int level)  ;
	void set_init_guess(void)  ;
  
	void calc_next_level_residual(int level) ;
	void zero_next_level(int level) ;
	void add_prolonged_prev_level(int level) ;
	void advance_in_time() ;
	double residual() { return  _residual ; }

	Tensor3d& P(int level=0) { return _P[level] ; } ;
  
	void prolong(Tensor3d& P, Tensor3d& pP) ;
	void restrict(Tensor3d& P, Tensor3d& rP) ;
  
private:

	double _residual;
	vector< vector< Tensor3d> > _G;
	vector<Tensor3d> _Div;
	vector<Tensor3d> _I;
	vector<Tensor3d> _P;
	double _dx, _dy , _dz;
	int _perms_num,_perms_len, _deg;
	vector< vector < vector< int > > > _perms;
  	int _len_z;
  
	typedef double* index;

	struct Tensore3dVI{
		void set(int len_x,int len_y,int len_z){
			VV.clear();
			VV.resize( len_x* len_y*len_z);
			_len_x=len_x;
			_len_y=len_y;
			_len_z=len_z;
		}
		vector<int>& operator()(int x,int y,int z){return VV[x+y*_len_x+z*_len_x*_len_y]; }
		vector<int>& operator[](int x){return VV[x]; }
		vector< vector<int> > VV;
		int _len_x;
		int _len_y;
		int _len_z;
	};

	struct Tensore3dI{
		void set(int len_x,int len_y,int len_z){
			VV.clear();
			VV.resize( len_x* len_y*len_z);
			_len_x=len_x;
			_len_y=len_y;
			_len_z=len_z;
		}
		int& operator()(int x,int y,int z){return VV[x+y*_len_x+z*_len_x*_len_y]; }
		vector<int>  VV;
		int _len_x;
		int _len_y;
		int _len_z;
	};
  
	struct Tensore3dVD{
		void set(int len_x,int len_y,int len_z){
			VV.clear();
			VV.resize( len_x* len_y*len_z);
			_len_x=len_x;
			_len_y=len_y;
			_len_z=len_z;
		}
		vector<float>& operator()(int x,int y,int z){return VV[x+y*_len_x+z*_len_x*_len_y]; }
		vector< vector<float> > VV;
		int _len_x;
		int _len_y;
		int _len_z;
	};
   
	struct Tensore3dVP{
		void set(int len_x,int len_y,int len_z){
			VV.clear();
			VV.resize( len_x* len_y*len_z);
			_len_x=len_x;
			_len_y=len_y;
			_len_z=len_z;
		}
		vector<pair<index,float> >& operator()(int x,int y,int z){return VV[x+y*_len_x+z*_len_x*_len_y]; }
		vector<pair<index,float> >& operator[](int x){return VV[x]; }
		vector< vector< pair<index,float> > > VV;
		int _len_x;
		int _len_y;
		int _len_z;
	};

	struct Tensore3dMD{
		Tensore3dMD(){
			VV=NULL;
		}

		~Tensore3dMD(){
			if (VV!=0)
				delete [] VV;
			VV=NULL;
		}

		void set(int len_x,int len_y,int len_z){
			if (VV!=0)
				delete [] VV;
			VV = new  vector< vector< double> >[len_x*len_y*len_z];
			_len_x=len_x;
			_len_y=len_y;
			_len_z=len_z;
		}
  
		vector< vector<double> >& operator()(int x,int y,int z){return VV[x+y*_len_x+z*_len_x*_len_y]; }
		vector< vector<double> >  *VV;
		int _len_x;
		int _len_y;
		int _len_z;
	};
  
	Tensore3dVP * _nebl;
} ;

#endif 
