export type LatLng = {
    latitude: number;
    longitude: number;
};

export type DistrictName =
    | 'Colombo'
    | 'Gampaha'
    | 'Kalutara'
    | 'Kandy'
    | 'Galle'
    | 'Matara'
    | 'Hambantota'
    | 'Anuradhapura'
    | 'Polonnaruwa'
    | 'Badulla'
    | 'NuwaraEliya'
    | 'Matale'
    | 'Jaffna';

/**
 * Simplified district boundary polygons
 * (mobile-friendly, visually correct)
 */
export const SRI_LANKA_DISTRICT_POLYGONS: Record<DistrictName, LatLng[]> = {
    Colombo: [
        { latitude: 6.8347, longitude: 79.8522 },
        { latitude: 6.8429, longitude: 79.8641 },
        { latitude: 6.8518, longitude: 79.8726 },
        { latitude: 6.8619, longitude: 79.8794 },
        { latitude: 6.8735, longitude: 79.8869 },
        { latitude: 6.8846, longitude: 79.8957 },
        { latitude: 6.8968, longitude: 79.9023 },
        { latitude: 6.9084, longitude: 79.9068 },
        { latitude: 6.9209, longitude: 79.9089 },
        { latitude: 6.9338, longitude: 79.9063 },
        { latitude: 6.9462, longitude: 79.9012 },
        { latitude: 6.9575, longitude: 79.8934 },
        { latitude: 6.9662, longitude: 79.8845 },
        { latitude: 6.9729, longitude: 79.8741 },
        { latitude: 6.9771, longitude: 79.8630 },
        { latitude: 6.9784, longitude: 79.8516 },
        { latitude: 6.9756, longitude: 79.8409 },
        { latitude: 6.9691, longitude: 79.8312 },
        { latitude: 6.9598, longitude: 79.8243 },
        { latitude: 6.9482, longitude: 79.8196 },
        { latitude: 6.9351, longitude: 79.8184 },
        { latitude: 6.9218, longitude: 79.8205 },
        { latitude: 6.9090, longitude: 79.8251 },
        { latitude: 6.8964, longitude: 79.8318 },
        { latitude: 6.8837, longitude: 79.8396 },
        { latitude: 6.8712, longitude: 79.8456 },
        { latitude: 6.8593, longitude: 79.8492 },
        { latitude: 6.8471, longitude: 79.8499 },
    ],


    Gampaha: [
        { latitude: 7.05, longitude: 79.80 },
        { latitude: 7.15, longitude: 79.88 },
        { latitude: 7.22, longitude: 79.92 },
        { latitude: 7.18, longitude: 79.85 },
        { latitude: 7.10, longitude: 79.78 },
    ],

    Kalutara: [
        { latitude: 6.55, longitude: 79.86 },
        { latitude: 6.65, longitude: 79.95 },
        { latitude: 6.72, longitude: 79.93 },
        { latitude: 6.66, longitude: 79.87 },
    ],

    Kandy: [
        { latitude: 7.20, longitude: 80.55 },
        { latitude: 7.30, longitude: 80.65 },
        { latitude: 7.35, longitude: 80.72 },
        { latitude: 7.25, longitude: 80.75 },
        { latitude: 7.18, longitude: 80.60 },
    ],

    NuwaraEliya: [
        { latitude: 6.85, longitude: 80.60 },
        { latitude: 6.95, longitude: 80.72 },
        { latitude: 6.98, longitude: 80.82 },
        { latitude: 6.88, longitude: 80.85 },
        { latitude: 6.82, longitude: 80.70 },
    ],

    Galle: [
    { latitude: 6.00, longitude: 80.15 },
    { latitude: 6.08, longitude: 80.25 },
    { latitude: 6.10, longitude: 80.35 },
    { latitude: 6.02, longitude: 80.32 },
    { latitude: 5.98, longitude: 80.20 },
],

    Matara: [
    { latitude: 5.90, longitude: 80.45 },
    { latitude: 6.00, longitude: 80.55 },
    { latitude: 6.02, longitude: 80.65 },
    { latitude: 5.95, longitude: 80.62 },
],

    Hambantota: [
    { latitude: 6.10, longitude: 81.05 },
    { latitude: 6.20, longitude: 81.15 },
    { latitude: 6.25, longitude: 81.25 },
    { latitude: 6.15, longitude: 81.22 },
],

    Anuradhapura: [
    { latitude: 8.20, longitude: 80.30 },
    { latitude: 8.35, longitude: 80.45 },
    { latitude: 8.40, longitude: 80.60 },
    { latitude: 8.25, longitude: 80.55 },
],

    Polonnaruwa: [
    { latitude: 7.90, longitude: 81.00 },
    { latitude: 8.05, longitude: 81.15 },
    { latitude: 8.10, longitude: 81.30 },
    { latitude: 7.95, longitude: 81.25 },
],

    Badulla: [
    { latitude: 6.90, longitude: 81.00 },
    { latitude: 7.00, longitude: 81.10 },
    { latitude: 7.05, longitude: 81.25 },
    { latitude: 6.95, longitude: 81.30 },
],

    Jaffna: [
    { latitude: 9.60, longitude: 79.90 },
    { latitude: 9.75, longitude: 80.00 },
    { latitude: 9.80, longitude: 80.10 },
    { latitude: 9.65, longitude: 80.15 },
],
    Matale: [
    { latitude: 9.60, longitude: 79.90 },
    { latitude: 9.75, longitude: 80.00 },
    { latitude: 9.80, longitude: 80.10 },
    { latitude: 9.65, longitude: 80.15 },
],
};
