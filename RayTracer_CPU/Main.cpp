#include <iostream>
#include <vector>
#include <cstring>
#include <algorithm> // std::sort
#include <omp.h> // parallel for
#include "GLFW/glfw3.h"
#include "glm.hpp"
#include "gtc/random.hpp"

#define WINDOW_WIDTH 768
#define WINDOW_HEIGHT 768
const int nNumSamples = 1000;
const int nMaxDepth = 5;
#define MEDIAN 5

#define EPSILON 0.0001f

struct Hit
{
    float t;
    bool m_bHit;
    glm::vec3 m_v3Pos;
    glm::vec3 m_v3Normal;
    glm::vec3 m_v3Color;
    bool m_bEmissive;
};

struct Ray 
{
    glm::vec3 m_v3Pos;
    glm::vec3 m_v3Dir;
};

class Object 
{
public:
    glm::vec3 m_v3Color;
    bool m_bEmissive;

    virtual Hit hit(Ray ray) const = 0;
};

class Triangle : public Object 
{
public:
    glm::vec3 v1, v2, v3;

    Hit hit(Ray ray) const 
    {
        Hit ret;
        ret.m_bHit = false;

        glm::vec3 v3Normal = glm::normalize(glm::cross(v2 - v1, v3 - v1));

        if (fabs(glm::dot(v3Normal, ray.m_v3Dir)) < 0.0f) 
        {
            return ret;
        }

        float t = (glm::dot(v3Normal, v1) - glm::dot(ray.m_v3Pos, v3Normal)) / glm::dot(ray.m_v3Dir, v3Normal);
        if (t < EPSILON) 
        {
            return ret;
        }

        glm::vec3 v3Pos = ray.m_v3Pos + (ray.m_v3Dir * t);

        glm::vec3 c1 = glm::cross(v2 - v1, v3Pos - v1);
        glm::vec3 c2 = glm::cross(v3 - v2, v3Pos - v2);
        glm::vec3 c3 = glm::cross(v1 - v3, v3Pos - v3);

        if (glm::dot(c1, v3Normal) < 0.0f || glm::dot(c2, v3Normal) < 0.0f || glm::dot(c3, v3Normal) < 0.0f)
        {
            return ret;
        }

        ret.m_bHit = true;
        ret.t = t;
        ret.m_v3Pos = v3Pos;
        ret.m_v3Normal = v3Normal;
        ret.m_v3Color = m_v3Color;
        ret.m_bEmissive = m_bEmissive;

        return ret;
    }
};

class Sphere : public Object
{
public:
    glm::vec3 m_v3Center;
    float m_fRadius;

    Hit hit(Ray ray) const
    {
        Hit ret;
        ret.m_bHit = false;

        glm::vec3 oc = ray.m_v3Pos - m_v3Center;
        float a = glm::dot(ray.m_v3Dir, ray.m_v3Dir);
        float b = glm::dot(oc, ray.m_v3Dir);
        float c = glm::dot(oc, oc) - m_fRadius * m_fRadius;

        float discriminant = b * b - a * c;

        if (discriminant > 0.0f)
        {
            float t1 = (-b - glm::sqrt(discriminant)) / a;
            float t2 = (-b + glm::sqrt(discriminant)) / a;

            if (t1 > EPSILON || t2 > EPSILON)
            {
                ret.m_bHit = true;
                ret.t = (t1 < t2) ? t1 : t2;
                ret.m_v3Pos = ray.m_v3Pos + ret.t * ray.m_v3Dir;
                ret.m_v3Normal = glm::normalize(ret.m_v3Pos - m_v3Center);
                ret.m_v3Color = m_v3Color;
                ret.m_bEmissive = m_bEmissive;

                return ret;
            }
        }

        return ret;
    }
};

float Random() 
{
    // [0.0f .. 1.0f]
    return ((float)rand() / (float)RAND_MAX);
}

glm::vec3 RandomVector() 
{
    glm::vec3 v3Dir = glm::sphericalRand(1.0f);
    return v3Dir;
}

glm::vec3 RandomDirection(glm::vec3 v3Normal) 
{
    return glm::normalize(RandomVector() + v3Normal);
}

class BitmapImage 
{
    void GetColor(int x, int y, int* pRed, int* pGreen, int* pBlue) 
    {
        *pRed   = data[(y * WINDOW_WIDTH * 3) + (x * 3) + 0];
        *pGreen = data[(y * WINDOW_WIDTH * 3) + (x * 3) + 1];
        *pBlue  = data[(y * WINDOW_WIDTH * 3) + (x * 3) + 2];
    }

    int GetMedian(std::vector<int> *pList) 
    {
        std::sort(pList->begin(), pList->end());

        int id = (MEDIAN * MEDIAN) / 2;
        return (*pList)[id];
    }

public:
    uint8_t* data;

    void Create() 
    {
        uint64_t nSize = WINDOW_WIDTH * WINDOW_HEIGHT * 3;
        data = new unsigned char[nSize];
        // Clear
        memset(data, 0, nSize);
    }

    void AddColor(int x, int y, glm::vec3 v3Color) 
    {
        int nRed, nGreen, nBlue;
        GetColor(x + 0, y + 0, &nRed, &nGreen, &nBlue);
        
        nRed   += (uint8_t)(v3Color.x * 255.0f);
        nGreen += (uint8_t)(v3Color.y * 255.0f);
        nBlue  += (uint8_t)(v3Color.z * 255.0f);

        data[(y * WINDOW_WIDTH * 3) + (x * 3) + 0] = (uint8_t)glm::clamp(nRed, 0, 255);
        data[(y * WINDOW_WIDTH * 3) + (x * 3) + 1] = (uint8_t)glm::clamp(nGreen, 0, 255);
        data[(y * WINDOW_WIDTH * 3) + (x * 3) + 2] = (uint8_t)glm::clamp(nBlue, 0, 255);
    }

    void MedianFilter() 
    {
        uint64_t nSize = WINDOW_WIDTH * WINDOW_HEIGHT * 3;
        uint8_t* dst = new unsigned char[nSize];
        // Clear
        memset(dst, 0, nSize);

        #pragma omp parallel for
        for (int x = 0; x < WINDOW_WIDTH; x++)
        {
            for (int y = 0; y < WINDOW_HEIGHT; y++) 
            {
                std::vector<int> listRed;
                std::vector<int> listGreen;
                std::vector<int> listBlue;

                for (int i = (-MEDIAN / 2); i <= (MEDIAN / 2); i++)
                {
                    for (int j = (-MEDIAN / 2); j <= (MEDIAN / 2); j++)
                    {
                        int nRed, nGreen, nBlue;

                        int x2 = x + i;
                        int y2 = y + j;
                        if (x2 < 0) { x2 = 0; }
                        if (x2 > (WINDOW_WIDTH - 1)) { x2 = WINDOW_WIDTH - 1; }
                        if (y2 < 0) { y2 = 0; }
                        if (y2 > (WINDOW_HEIGHT - 1)) { y2 = WINDOW_HEIGHT - 1; }

                        GetColor(x2, y2, &nRed, &nGreen, &nBlue);
                        listRed.push_back(nRed);
                        listGreen.push_back(nGreen);
                        listBlue.push_back(nBlue);
                    }
                }

                int nNewRed = GetMedian(&listRed);
                int nNewGreen = GetMedian(&listGreen);
                int nNewBlue = GetMedian(&listBlue);
                
                dst[(y * WINDOW_WIDTH * 3) + (x * 3) + 0] = (uint8_t)nNewRed;
                dst[(y * WINDOW_WIDTH * 3) + (x * 3) + 1] = (uint8_t)nNewGreen;
                dst[(y * WINDOW_WIDTH * 3) + (x * 3) + 2] = (uint8_t)nNewBlue;
            }
        }

        // apply
        data = dst;
    }
};

BitmapImage bitmap;
std::vector< Object* > objects;
glm::vec3 v3Eye = glm::vec3(0, 0, 5.0f);
const float fBrightness = (2.0f * 3.141592654f) * (1.0f / float(nNumSamples));


glm::vec3 PathTrace(Ray ray, int nDepth) 
{
    if (nDepth > nMaxDepth) 
    {
        return glm::vec3(0, 0, 0);
    }

    float minT = 1000000.0f;
    Hit hi, finalHi;
    finalHi.m_bHit = false;

    for (int i = 0; i < (int)objects.size(); i++)
    {
        hi = objects[i]->hit(ray);
        if (true == hi.m_bHit && hi.t < minT)
        {
            finalHi = hi;
            minT = hi.t;
        }
    }

    if (false == finalHi.m_bHit) 
    {
        return glm::vec3(0, 0, 0);
    }

    if (true == finalHi.m_bEmissive) 
    {
        return finalHi.m_v3Color;
    }

    float fDiffuseIntensity = glm::dot(-ray.m_v3Dir, finalHi.m_v3Normal);
    if (fDiffuseIntensity <= 0.0f) 
    {
        return glm::vec3(0, 0, 0);
    }

    Ray randomRay;
    randomRay.m_v3Pos = finalHi.m_v3Pos + (finalHi.m_v3Normal * 0.001f);
    randomRay.m_v3Dir = RandomDirection(finalHi.m_v3Normal);

    glm::vec3 v3OriginalColor = fDiffuseIntensity * finalHi.m_v3Color;
    glm::vec3 v3NewColor = PathTrace(randomRay, nDepth + 1);
    glm::vec3 v3Color = v3OriginalColor * v3NewColor;

    return v3Color;
}

void Init()
{
    Sphere* pSphere = new Sphere();
    pSphere->m_v3Center = glm::vec3(0.0f, -0.7f, -0.5f);
    pSphere->m_fRadius = 0.3f;
    pSphere->m_v3Color = glm::vec3(2, 2, 2);
    pSphere->m_bEmissive = false;
    objects.push_back(pSphere);

    Triangle *pFloor1 = new Triangle();
    pFloor1->v1 = glm::vec3(1.0f, -1.0f, 1.0f);
    pFloor1->v2 = glm::vec3(-1.0f, -1.0f, -1.0f);
    pFloor1->v3 = glm::vec3(-1.0f, -1.0f, 1.0f);
    pFloor1->m_v3Color = glm::vec3(1, 1, 1);
    pFloor1->m_bEmissive = false;
    objects.push_back(pFloor1);

    Triangle* pFloor2 = new Triangle();
    pFloor2->v1 = glm::vec3(1.0f, -1.0f, 1.0f);
    pFloor2->v2 = glm::vec3(1.0f, -1.0f, -1.0f);
    pFloor2->v3 = glm::vec3(-1.0f, -1.0f, -1.0f);
    pFloor2->m_v3Color = glm::vec3(1, 1, 1);
    pFloor2->m_bEmissive = false;
    objects.push_back(pFloor2);

    Triangle* pLight1 = new Triangle();
    pLight1->v1 = glm::vec3(-0.5f, 0.99f, 0.5f);
    pLight1->v2 = glm::vec3(-0.5f, 0.99f, -0.5f);
    pLight1->v3 = glm::vec3(0.5f, 0.99f, 0.5f);
    pLight1->m_v3Color = glm::vec3(1, 1, 1);
    pLight1->m_bEmissive = true;
    objects.push_back(pLight1);

    Triangle* pLight2 = new Triangle();   
    pLight2->v1 = glm::vec3(-0.5f, 0.99f, -0.5f);
    pLight2->v2 = glm::vec3(0.5f, 0.99f, -0.5f);
    pLight2->v3 = glm::vec3(0.5f, 0.99f, 0.5f);
    pLight2->m_v3Color = glm::vec3(1, 1, 1);
    pLight2->m_bEmissive = true;
    objects.push_back(pLight2);

    Triangle* pCeiling1 = new Triangle();
    pCeiling1->v1 = glm::vec3(-1, 1, 1);
    pCeiling1->v2 = glm::vec3(-1,1,-1);
    pCeiling1->v3 = glm::vec3(1, 1, 1);
    pCeiling1->m_v3Color = glm::vec3(1, 1, 1);
    pCeiling1->m_bEmissive = false;
    objects.push_back(pCeiling1);

    Triangle* pCeiling2 = new Triangle();
    pCeiling2->v1 = glm::vec3(-1, 1, -1);
    pCeiling2->v2 = glm::vec3(1, 1, -1);
    pCeiling2->v3 = glm::vec3(1, 1, 1);
    pCeiling2->m_v3Color = glm::vec3(1, 1, 1);
    pCeiling2->m_bEmissive = false;
    objects.push_back(pCeiling2);

    Triangle* pLeft1 = new Triangle();
    pLeft1->v1 = glm::vec3(-1,-1,-1);
    pLeft1->v2 = glm::vec3(-1,1,1);
    pLeft1->v3 = glm::vec3(-1,-1,1);
    pLeft1->m_v3Color = glm::vec3(1, 0, 0);
    pLeft1->m_bEmissive = false;
    objects.push_back(pLeft1);

    Triangle* pLeft2 = new Triangle();
    pLeft2->v1 = glm::vec3(-1,-1,-1);
    pLeft2->v2 = glm::vec3(-1,1,-1);
    pLeft2->v3 = glm::vec3(-1,1,1);
    pLeft2->m_v3Color = glm::vec3(1, 0, 0);
    pLeft2->m_bEmissive = false;
    objects.push_back(pLeft2);

    Triangle* pRight1 = new Triangle();
    pRight1->v1 = glm::vec3(1,1,1);
    pRight1->v2 = glm::vec3(1,-1,-1);
    pRight1->v3 = glm::vec3(1,-1,1);
    pRight1->m_v3Color = glm::vec3(0, 1, 0);
    pRight1->m_bEmissive = false;
    objects.push_back(pRight1);

    Triangle* pRight2 = new Triangle();
    pRight2->v1 = glm::vec3(1,-1,-1);
    pRight2->v2 = glm::vec3(1,1,1);
    pRight2->v3 = glm::vec3(1,1,-1);
    pRight2->m_v3Color = glm::vec3(0, 1, 0);
    pRight2->m_bEmissive = false;
    objects.push_back(pRight2);

    Triangle* pBack1 = new Triangle();
    pBack1->v1 = glm::vec3(1,-1,-1);
    pBack1->v2 = glm::vec3(-1,1,-1);
    pBack1->v3 = glm::vec3(-1,-1,-1);
    pBack1->m_v3Color = glm::vec3(1, 1, 1);
    pBack1->m_bEmissive = false;
    objects.push_back(pBack1);

    Triangle* pBack2 = new Triangle();
    pBack2->v1 = glm::vec3(1,-1,-1);
    pBack2->v2 = glm::vec3(1,1,-1);
    pBack2->v3 = glm::vec3(-1,1,-1);
    pBack2->m_v3Color = glm::vec3(1, 1, 1);
    pBack2->m_bEmissive = false;
    objects.push_back(pBack2);

    bitmap.Create();
}

void Update() 
{
    #pragma omp parallel for
    for (int y = 0; y < WINDOW_HEIGHT; y++)
    {
        for (int x = 0; x < WINDOW_WIDTH; x++)
        {
            Hit hi, finalHi;
            Ray ray;
            float i = 0.0f, j = 0.0f, minT = 1000000.0f;
            glm::vec3 v3Color = glm::vec3(0, 0, 0);

            minT = 1000000.0f;
            finalHi.m_bHit = false;
            hi.m_bHit = false;
            v3Color = glm::vec3(0, 0, 0);

            i = (2.0f * float(x) / float(WINDOW_WIDTH)) - 1.0f;
            j = (2.0f * float(y) / float(WINDOW_HEIGHT)) - 1.0f;

            ray.m_v3Pos = glm::vec3(i, j, 1.2f);
            ray.m_v3Dir = glm::normalize(ray.m_v3Pos - v3Eye);

            for (int k = 0; k < (int)objects.size(); k++)
            {
                hi = objects[k]->hit(ray);
                if (true == hi.m_bHit && hi.t < minT)
                {
                    finalHi = hi;
                    minT = hi.t;
                }
            }

            if (true == finalHi.m_bHit)
            {
                if (true == finalHi.m_bEmissive)
                {
                    v3Color = finalHi.m_v3Color;
                }
                else
                {
                    Ray randomRay;
                    randomRay.m_v3Pos = finalHi.m_v3Pos;
                    randomRay.m_v3Dir = RandomDirection(finalHi.m_v3Normal);

                    glm::vec3 v3OriginalColor = finalHi.m_v3Color;
                    glm::vec3 v3NewColor = PathTrace(randomRay, 0);
                    v3Color = v3OriginalColor * v3NewColor;

                    v3Color *= fBrightness;
                }
            }

            bitmap.AddColor(x, y, v3Color);
        }
    }
}

void Draw() 
{
    glClearColor(1.0f, 1.0f, 1.0f, 1.0f);
    glClear(GL_COLOR_BUFFER_BIT);

    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    glOrtho(0, WINDOW_WIDTH, 0, WINDOW_HEIGHT, -1, 1);

    glMatrixMode(GL_MODELVIEW);
    glLoadIdentity();

    glRasterPos2i(0, 0);
    glDrawPixels(WINDOW_WIDTH, WINDOW_HEIGHT, GL_RGB, GL_UNSIGNED_BYTE, bitmap.data);
}

int main() 
{
    GLFWwindow* window;

    if (!glfwInit()) 
    {
        std::cerr << "Failed to initialize GLFW" << std::endl;
        return -1;
    }

    glfwWindowHint(GLFW_RESIZABLE, GLFW_FALSE); // no resize window
    glfwSwapInterval(1); // disable vsync

    window = glfwCreateWindow(WINDOW_WIDTH, WINDOW_HEIGHT, "RayTracer (CPU Version)", NULL, NULL);
    if (!window) {
        glfwTerminate();
        return -1;
    }

    glfwMakeContextCurrent(window);

    Init();

    float s = 0.0f;

    int nElapsedPercent = 0;
    int nPercent = 0;

    while (false == glfwWindowShouldClose(window)) 
    {
        nPercent = (int)(((float)s / (float)nNumSamples) * 100.0f);
        if (nElapsedPercent != nPercent) 
        {
            nElapsedPercent = nPercent;

            char strTitle[100];
            std::sprintf(strTitle, "RayTracer (CPU Version) - %d%%", nPercent);
            glfwSetWindowTitle(window, strTitle);
        }

        if (s < nNumSamples)
        {
            Update();
            s++;
        }
        else
        {
            static bool bFirst = true;

            if (true == bFirst) 
            {
                for (int i = 0; i < 10; i++) 
                {
                    bitmap.MedianFilter();
                }
                bFirst = false;
            }
        }
        Draw();

        glfwSwapBuffers(window);
        glfwPollEvents();
    }

    glfwTerminate();
    return 0;
}