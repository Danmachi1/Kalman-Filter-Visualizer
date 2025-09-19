#include "uNavINS.h"
#include <new>   // placement new
#include <cmath> // optional

extern "C" {

// Use static buffer and placement new
static uNavINS filter;

// Reconstruct filter
void unavins_init() {
    filter.~uNavINS();
    new (&filter) uNavINS();
}

// Call uNavINS::update(...)
void unavins_update(float ax, float ay, float az,
                    float gx, float gy, float gz,
                    float lat, float lon, float alt) {
    filter.update(
        0, 0.0, 0.0, 0.0,
        static_cast<double>(lat),
        static_cast<double>(lon),
        static_cast<double>(alt),
        gx, gy, gz,
        ax, ay, az,
        0.0f, 0.0f, 0.0f
    );
}


// Provide dummy dt-based wrapper (optional)
void unavins_predict(float dt) {
    // uNavINS does not expose predict-only
    // This function is a placeholder
}

// Access public getters
float unavins_getVn()  { return filter.getVelN(); }
float unavins_getVe()  { return filter.getVelE(); }
float unavins_getVd()  { return filter.getVelD(); }

double unavins_getLat() { return filter.getLatitude_rad(); }
double unavins_getLon() { return filter.getLongitude_rad(); }
double unavins_getAlt() { return filter.getAltitude_m(); }

float unavins_getHeading() { return filter.getHeading_rad(); }
float unavins_getRoll()    { return filter.getRoll_rad(); }
float unavins_getPitch()   { return filter.getPitch_rad(); }

}
