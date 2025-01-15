# 2. Geometry and Transformations
2.1 대칭성을 이용해 축 정렬 경계 상자를 변환하는 좀 더 효율적인 방법을 찾아보자. 8개의 꼭짓점은 세 개의 축 정렬 기저 벡터와 한 꼭지점의 선형 조합이므로, 변환된 경계 상자는 여기서 소개한 방법보다 훨씬 효율적으로 찾을 수 있다. (Arvo 1990).  
Sol : 문제에 표시된 [Arvo, J. 1990. Transforming axis-aligned bounding boxes. In A. S. Glassner (Ed.), Graphics Gems I, 548–50. San Diego: Academic Press.](http://inis.jinr.ru/sl/vol1/CMC/Graphics_Gems_1,ed_A.Glassner.pdf) 에서 아이디어에 대한 지식을 얻을 수 있다.  
위 링크를 통해 해당 section을 읽어보면 아래와 같은 내용을 확인할 수 있다.  

> We address two common methods of encoding a bounding box, B. The first is the use of three intervals, $(B^{min}_{x}, B^{max}_{x}), (B^{min}_{y}, B^{max}_{y}), (B^{min}_{z}, B^{max}_{z})$. Aside from ordering, this is equivalent to storing two opposing vertices. The second method is to store the box center, $(B^{cent}_{x}, B^{cent}_{y}, B^{cent}_{z})$, and the box half-diagonal, $(B^{diag}_{x}, B^{diag}_{y}, B^{diag}_{z})$, which is the positive vector from the center of the box to the vertex with the three largest components.  

위와 같이 언급이 되어 있기 때문에 우리는 AABB의 대칭성을 이용해서 Bounding Volume 을 표현할 수 있다.   
책에서는 아래와 같이 procedure 를 나타내고 있다. (책 내용이 아닌 내가 이해한 것을 C++로 대략 작성했다.)
```cpp
func Transform_CenterDiag_Box(M,T,A,B)
{
    for (int axis=0; axis < 3; axis++)
        // init the output variable and setting the new center 
        B.cent[axis] = T[i];
        B.diag[axis] = 0;

        // Compute the i'th coordinate of the center by adding M, A.cent
        // and the i'th coordinate of the half-diagonal by add |M|, A.diag
        for (int j=0; j<3; j++)
            B.cent[i] = B.cent[i] + M[axis][j] * A.cent[j];
            B.diag[i] = B.diag[i] + |M[axis][j]| * A.diag[j];
        end
    end;
}
```  
이제 위의 코드를 이해해보자. 위의 코드에서 사용된 것은 figure 설명에 추가되어 있다.  
> Figure 2. An algorithm for transforming an axis-aligned bounding box, A, stored as a center and a half-diagonal into another box, B, Matrix, M, translate, T.  

일단 까마득 하지만 풀어봤다.
```cpp
// hyeon 
template <typename T>
inline Bounds3<T> Transform::operator()(const Bounds3<T> &A) const {
    Bounds3<T> B; // return value
    Vector3<T> B_diag; Point3<T> B_cent; // 계산을 위해서 저장할 변수

    // this M matrix includes Rotation matrix and Translation matrix.
    // Matrix size 는 4 x 4이기 때문에 rotation 과 translation 이 모두 합쳐짐.
    // 분리할 필요 존재.
    const Matrix4x4 &M = this->GetMatrix();

    // center point 와 diag vector 추출
    Vector3<T> A_diag = A.Diagonal(); // diag of A
    Point3<T> A_cent = (A.pMax + A.pMin) * 0.5; // center point of A

    // (Arvo 1990). 에 나온 방법 활용
    // i = axis and row 
    for (int i = 0; i < 3; i++) {

        B_cent[i] = M.m[i][3];
        B_diag[i] = 0;

        for (int j = 0; j < 3; j++) {
            B_cent[i] += M.m[i][j] * A_cent[j];
            B_diag[i] += std::abs(M.m[i][j]) * A_diag[j];
        }
    }

    // 저장을 위해서 다시 min max 좌표로 변경
    Vector3<T> B_diag_div_2 = B_diag * 0.5;
    B.pMin = B_cent - B_diag_div_2;
    B.pMax = B_cent + B_diag_div_2;

    // min, max가 만약 바뀌었다면 swap 
    if (B.pMax < B.pMin) {
        swap(B.pMax, B.pMin);
    }

    // result return
    return B;
}
```
주석을 참고하면 설명을 이어나갈 수 있다. Unit test Code 도 pbrt에서 지원 해준다.  
한번 수행해보는 것도 좋다.  
(역시 엄청 어렵게 느껴진다.)

2.2 상자 대신 많은 수직하지 않는 조각의 교차점을 이용해서 물체 주변에 좀 더 달라붙은(tighter) 경계를 계산할 수 있다. 사용자에게 임의의 조각들로 구성된 경계를 명시할 수 있도록 pbrt 경계 상자 클래스를 확장하라.    
Sol : Bounding Volume Hierarchy를 구축할 때 AABB 뿐만아니라 Sphere, Object를 중심으로 하는 bounding box 등 여러 타입이 존재한다. 이에 따라서 구축을 진행하게 된다면 아래와 같이 확장 할 수 있다. 


2.3 ```Normal3f```를 ```Vector3f```처럼 변환하도록 pbrt를 변경한 뒤 해당 버그로 인한 잘못된 결과를 명확히 보여주는 장면을 생성하라 (끝난 뒤에 소스코드의 변경을 제거하는 것을 잊지 마라!).  
Sol : 2.8.3 Sec

2.4 예를 들어 변환의 이동 요소만 시간에 다라 변할 경우 ```AnimationTransform``` 의 구현은 같은 두 회전 간의 필요 없는 보간을 계산하게 된다. ```AnimatedTransform```의 구현을 현재 구현의 범용성이 필요없을 경우 이런 필요 없는 계산을 회피하도록 변경하라. 최적화로 지원되는 장면에 대해 성능이 얼마나 많이 향상됐는가?  