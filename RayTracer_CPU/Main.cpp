#include <iostream>
#include <vector>
#include <cstring>
#include <algorithm>
#include <omp.h>
#include "GLFW/glfw3.h"
#include "glm.hpp"
#include "gtc/random.hpp"

#define WINDOW_WIDTH 720
#define WINDOW_HEIGHT 720
const int nNumSamples = 20;
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
    bool m_bSpecular;      // új: specular flag

    Object() : m_bEmissive(false), m_bSpecular(false) {}

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
            return ret;

        float t = (glm::dot(v3Normal, v1) - glm::dot(ray.m_v3Pos, v3Normal)) /
            glm::dot(ray.m_v3Dir, v3Normal);
        if (t < EPSILON)
            return ret;

        glm::vec3 v3Pos = ray.m_v3Pos + (ray.m_v3Dir * t);

        glm::vec3 c1 = glm::cross(v2 - v1, v3Pos - v1);
        glm::vec3 c2 = glm::cross(v3 - v2, v3Pos - v2);
        glm::vec3 c3 = glm::cross(v1 - v3, v3Pos - v3);

        if (glm::dot(c1, v3Normal) < 0.0f ||
            glm::dot(c2, v3Normal) < 0.0f ||
            glm::dot(c3, v3Normal) < 0.0f)
            return ret;

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
        *pRed = data[(y * WINDOW_WIDTH * 3) + (x * 3) + 0];
        *pGreen = data[(y * WINDOW_WIDTH * 3) + (x * 3) + 1];
        *pBlue = data[(y * WINDOW_WIDTH * 3) + (x * 3) + 2];
    }

    int GetMedian(std::vector<int>* pList)
    {
        std::sort(pList->begin(), pList->end());
        int id = (MEDIAN * MEDIAN) / 2;
        return (*pList)[id];
    }

public:
    uint8_t* data;
    float* accum;

    void Create()
    {
        uint64_t nSize = WINDOW_WIDTH * WINDOW_HEIGHT * 3;
        data = new unsigned char[nSize];
        memset(data, 0, nSize);

        accum = new float[nSize];
        memset(accum, 0, nSize * sizeof(float));
    }

    void AddSample(int x, int y, glm::vec3 v3Color)
    {
        int idx = (y * WINDOW_WIDTH * 3) + (x * 3);
        accum[idx + 0] += glm::clamp(v3Color.x, 0.0f, 1.0f);
        accum[idx + 1] += glm::clamp(v3Color.y, 0.0f, 1.0f);
        accum[idx + 2] += glm::clamp(v3Color.z, 0.0f, 1.0f);
    }

    void Finalize()
    {
        uint64_t nSize = WINDOW_WIDTH * WINDOW_HEIGHT * 3;
        for (uint64_t i = 0; i < nSize; ++i)
        {
            float v = accum[i] / float(nNumSamples);
            v = glm::clamp(v, 0.0f, 1.0f);
            data[i] = (uint8_t)(v * 255.0f);
        }
    }

    void MedianFilter()
    {
        uint64_t nSize = WINDOW_WIDTH * WINDOW_HEIGHT * 3;
        uint8_t* dst = new unsigned char[nSize];
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
                        if (x2 < 0) x2 = 0;
                        if (x2 > (WINDOW_WIDTH - 1)) x2 = WINDOW_WIDTH - 1;
                        if (y2 < 0) y2 = 0;
                        if (y2 > (WINDOW_HEIGHT - 1)) y2 = WINDOW_HEIGHT - 1;

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

        data = dst;
    }
};

BitmapImage bitmap;
std::vector<Object*> objects;
glm::vec3 v3Eye = glm::vec3(0, 0, 5.0f);

float LightFalloff = 2.0f;   // állítható fényerő-esés

glm::vec3 DirectLight(const Hit& hit)
{
    glm::vec3 result(0.0f);

    glm::vec3 brdf = hit.m_v3Color / 3.14159265f;

    for (auto obj : objects)
    {
        if (!obj->m_bEmissive) continue;

        Triangle* tri = dynamic_cast<Triangle*>(obj);
        if (!tri) continue;

        float r1 = Random();
        float r2 = Random();
        float sqrtR1 = sqrt(r1);

        glm::vec3 lightPos =
            tri->v1 * (1.0f - sqrtR1) +
            tri->v2 * (sqrtR1 * (1.0f - r2)) +
            tri->v3 * (sqrtR1 * r2);

        glm::vec3 L = glm::normalize(lightPos - hit.m_v3Pos);

        float dist = glm::length(lightPos - hit.m_v3Pos);
        float falloff = LightFalloff / (dist * dist);

        Ray shadow;
        shadow.m_v3Pos = hit.m_v3Pos + hit.m_v3Normal * 0.001f;
        shadow.m_v3Dir = L;

        float minT = 1000000.0f;
        Hit h2;
        bool blocked = false;

        for (auto o : objects)
        {
            h2 = o->hit(shadow);
            if (h2.m_bHit && h2.t < minT)
            {
                minT = h2.t;
                if (!o->m_bEmissive)
                    blocked = true;
            }
        }

        if (!blocked)
        {
            float cosTheta = glm::dot(hit.m_v3Normal, L);
            if (cosTheta > 0.0f)
                result += obj->m_v3Color * brdf * cosTheta * falloff;
        }
    }

    return result;
}

glm::vec3 PathTrace(Ray ray, int nDepth)
{
    if (nDepth > nMaxDepth)
        return glm::vec3(0, 0, 0);

    float minT = 1000000.0f;
    Hit hi, finalHi;
    finalHi.m_bHit = false;
    Object* hitObj = nullptr;

    for (auto obj : objects)
    {
        hi = obj->hit(ray);
        if (hi.m_bHit && hi.t < minT)
        {
            finalHi = hi;
            minT = hi.t;
            hitObj = obj;
        }
    }

    if (!finalHi.m_bHit || hitObj == nullptr)
        return glm::vec3(0, 0, 0);

    if (finalHi.m_bEmissive)
        return finalHi.m_v3Color;

    // SPECULAR (tükör / fényes felület)
    if (hitObj->m_bSpecular)
    {
        // egyszerű keverés: 20% specular, 80% diffuse+indirect
        float specAmount = 0.2f;

        // specular komponens
        glm::vec3 reflectDir = glm::reflect(ray.m_v3Dir, finalHi.m_v3Normal);
        Ray reflectRay;
        reflectRay.m_v3Pos = finalHi.m_v3Pos + finalHi.m_v3Normal * 0.001f;
        reflectRay.m_v3Dir = glm::normalize(reflectDir);

        glm::vec3 specColor = PathTrace(reflectRay, nDepth + 1);

        // diffuse + indirect
        glm::vec3 albedo = finalHi.m_v3Color;
        glm::vec3 direct = DirectLight(finalHi);

        Ray randomRay;
        randomRay.m_v3Pos = finalHi.m_v3Pos + finalHi.m_v3Normal * 0.001f;
        randomRay.m_v3Dir = RandomDirection(finalHi.m_v3Normal);

        float cosTheta = glm::dot(finalHi.m_v3Normal, randomRay.m_v3Dir);
        glm::vec3 indirect(0.0f);
        if (cosTheta > 0.0f)
        {
            glm::vec3 brdf = albedo / 3.14159265f;
            glm::vec3 incoming = PathTrace(randomRay, nDepth + 1);
            indirect = brdf * incoming * cosTheta;
        }

        glm::vec3 diffuseTotal = direct + indirect;
        return (1.0f - specAmount) * diffuseTotal + specAmount * specColor;
    }

    // DIFFUSE (Lambert)
    glm::vec3 albedo = finalHi.m_v3Color;

    glm::vec3 direct = DirectLight(finalHi);

    Ray randomRay;
    randomRay.m_v3Pos = finalHi.m_v3Pos + finalHi.m_v3Normal * 0.001f;
    randomRay.m_v3Dir = RandomDirection(finalHi.m_v3Normal);

    float cosTheta = glm::dot(finalHi.m_v3Normal, randomRay.m_v3Dir);
    if (cosTheta <= 0.0f)
        return direct;

    glm::vec3 brdf = albedo / 3.14159265f;

    glm::vec3 incoming = PathTrace(randomRay, nDepth + 1);
    glm::vec3 indirect = brdf * incoming * cosTheta;

    return direct + indirect;
}

void Init()
{
    Sphere* pSphere = new Sphere();
    pSphere->m_v3Center = glm::vec3(0.0f, -0.7f, -0.5f);
    pSphere->m_fRadius = 0.3f;
    pSphere->m_v3Color = glm::vec3(0.8f, 0.8f, 0.8f);
    pSphere->m_bEmissive = false;
    pSphere->m_bSpecular = true; // gömb: fényes / tükröződő
    objects.push_back(pSphere);

    Triangle* pFloor1 = new Triangle();
    pFloor1->v1 = glm::vec3(1.0f, -1.0f, 1.0f);
    pFloor1->v2 = glm::vec3(-1.0f, -1.0f, -1.0f);
    pFloor1->v3 = glm::vec3(-1.0f, -1.0f, 1.0f);
    pFloor1->m_v3Color = glm::vec3(0.7f, 0.7f, 0.7f);
    pFloor1->m_bEmissive = false;
    pFloor1->m_bSpecular = false;
    objects.push_back(pFloor1);

    Triangle* pFloor2 = new Triangle();
    pFloor2->v1 = glm::vec3(1.0f, -1.0f, 1.0f);
    pFloor2->v2 = glm::vec3(1.0f, -1.0f, -1.0f);
    pFloor2->v3 = glm::vec3(-1.0f, -1.0f, -1.0f);
    pFloor2->m_v3Color = glm::vec3(0.7f, 0.7f, 0.7f);
    pFloor2->m_bEmissive = false;
    pFloor2->m_bSpecular = false;
    objects.push_back(pFloor2);

    Triangle* pLight1 = new Triangle();
    pLight1->v1 = glm::vec3(-0.5f, 0.99f, 0.5f);
    pLight1->v2 = glm::vec3(-0.5f, 0.99f, -0.5f);
    pLight1->v3 = glm::vec3(0.5f, 0.99f, 0.5f);
    pLight1->m_v3Color = glm::vec3(1.5f, 1.5f, 1.5f);
    pLight1->m_bEmissive = true;
    pLight1->m_bSpecular = false;
    objects.push_back(pLight1);

    Triangle* pLight2 = new Triangle();
    pLight2->v1 = glm::vec3(-0.5f, 0.99f, -0.5f);
    pLight2->v2 = glm::vec3(0.5f, 0.99f, -0.5f);
    pLight2->v3 = glm::vec3(0.5f, 0.99f, 0.5f);
    pLight2->m_v3Color = glm::vec3(1.5f, 1.5f, 1.5f);
    pLight2->m_bEmissive = true;
    pLight2->m_bSpecular = false;
    objects.push_back(pLight2);

    Triangle* pCeiling1 = new Triangle();
    pCeiling1->v1 = glm::vec3(-1, 1, 1);
    pCeiling1->v2 = glm::vec3(-1, 1, -1);
    pCeiling1->v3 = glm::vec3(1, 1, 1);
    pCeiling1->m_v3Color = glm::vec3(0.7f, 0.7f, 0.7f);
    pCeiling1->m_bEmissive = false;
    pCeiling1->m_bSpecular = false;
    objects.push_back(pCeiling1);

    Triangle* pCeiling2 = new Triangle();
    pCeiling2->v1 = glm::vec3(-1, 1, -1);
    pCeiling2->v2 = glm::vec3(1, 1, -1);
    pCeiling2->v3 = glm::vec3(1, 1, 1);
    pCeiling2->m_v3Color = glm::vec3(0.7f, 0.7f, 0.7f);
    pCeiling2->m_bEmissive = false;
    pCeiling2->m_bSpecular = false;
    objects.push_back(pCeiling2);

    Triangle* pLeft1 = new Triangle();
    pLeft1->v1 = glm::vec3(-1, -1, -1);
    pLeft1->v2 = glm::vec3(-1, 1, 1);
    pLeft1->v3 = glm::vec3(-1, -1, 1);
    pLeft1->m_v3Color = glm::vec3(0.75f, 0.1f, 0.1f);
    pLeft1->m_bEmissive = false;
    pLeft1->m_bSpecular = false;
    objects.push_back(pLeft1);

    Triangle* pLeft2 = new Triangle();
    pLeft2->v1 = glm::vec3(-1, -1, -1);
    pLeft2->v2 = glm::vec3(-1, 1, -1);
    pLeft2->v3 = glm::vec3(-1, 1, 1);
    pLeft2->m_v3Color = glm::vec3(0.75f, 0.1f, 0.1f);
    pLeft2->m_bEmissive = false;
    pLeft2->m_bSpecular = false;
    objects.push_back(pLeft2);

    Triangle* pRight1 = new Triangle();
    pRight1->v1 = glm::vec3(1, 1, 1);
    pRight1->v2 = glm::vec3(1, -1, -1);
    pRight1->v3 = glm::vec3(1, -1, 1);
    pRight1->m_v3Color = glm::vec3(0.1f, 0.75f, 0.1f);
    pRight1->m_bEmissive = false;
    pRight1->m_bSpecular = false;
    objects.push_back(pRight1);

    Triangle* pRight2 = new Triangle();
    pRight2->v1 = glm::vec3(1, -1, -1);
    pRight2->v2 = glm::vec3(1, 1, 1);
    pRight2->v3 = glm::vec3(1, 1, -1);
    pRight2->m_v3Color = glm::vec3(0.1f, 0.75f, 0.1f);
    pRight2->m_bEmissive = false;
    pRight2->m_bSpecular = false;
    objects.push_back(pRight2);

    Triangle* pBack1 = new Triangle();
    pBack1->v1 = glm::vec3(1, -1, -1);
    pBack1->v2 = glm::vec3(-1, 1, -1);
    pBack1->v3 = glm::vec3(-1, -1, -1);
    pBack1->m_v3Color = glm::vec3(0.7f, 0.7f, 0.7f);
    pBack1->m_bEmissive = false;
    pBack1->m_bSpecular = false;
    objects.push_back(pBack1);

    Triangle* pBack2 = new Triangle();
    pBack2->v1 = glm::vec3(1, -1, -1);
    pBack2->v2 = glm::vec3(1, 1, -1);
    pBack2->v3 = glm::vec3(-1, 1, -1);
    pBack2->m_v3Color = glm::vec3(0.7f, 0.7f, 0.7f);
    pBack2->m_bEmissive = false;
    pBack2->m_bSpecular = false;
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
            Ray ray;

            float i = (2.0f * float(x) / float(WINDOW_WIDTH)) - 1.0f;
            float j = (2.0f * float(y) / float(WINDOW_HEIGHT)) - 1.0f;

            ray.m_v3Pos = glm::vec3(i, j, 1.2f);
            ray.m_v3Dir = glm::normalize(ray.m_v3Pos - v3Eye);

            glm::vec3 sample = PathTrace(ray, 0);
            bitmap.AddSample(x, y, sample);
        }
    }

    bitmap.Finalize();
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

    glfwWindowHint(GLFW_RESIZABLE, GLFW_FALSE);
    glfwSwapInterval(1);

    window = glfwCreateWindow(WINDOW_WIDTH, WINDOW_HEIGHT, "RayTracer (CPU BRDF + Specular Averaged)", NULL, NULL);
    if (!window)
    {
        glfwTerminate();
        return -1;
    }

    glfwMakeContextCurrent(window);

    Init();

    float s = 0.0f;
    int nElapsedPercent = 0;
    int nPercent = 0;

    while (!glfwWindowShouldClose(window))
    {
        nPercent = (int)(((float)s / (float)nNumSamples) * 100.0f);
        if (nElapsedPercent != nPercent)
        {
            nElapsedPercent = nPercent;
            char strTitle[100];
            std::sprintf(strTitle, "RayTracer (CPU BRDF + Specular Averaged) - %d%%", nPercent);
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
            if (bFirst)
            {
                bitmap.Finalize();
                // opcionális zajcsökkentés:
                // for (int i = 0; i < 10; i++)
                //     bitmap.MedianFilter();
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