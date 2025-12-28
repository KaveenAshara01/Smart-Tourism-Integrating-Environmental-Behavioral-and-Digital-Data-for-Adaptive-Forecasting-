// utils/mockData.ts
import { DistrictName } from './sriLankaDistrictPolygons';

export interface DistrictData {
    id: string;
    name: DistrictName;
    coordinates: { latitude: number; longitude: number };
    bounds: number[][];  // [lat, lon] polygon for highlighting
    currentStats: {
        touristCount: number;
        trafficLevel: number;  // 0-100
        weatherTemp: number;
        weatherCondition: string;
        safetyScore: number;  // 0-100
        crisisLevel: 'low' | 'medium' | 'high';
        topAttractions: string[];
    };
    forecast: {
        nextMonth: number;  // Predicted tourist arrivals
        confidence: number;
    };
}

export const DISTRICT_DATA: DistrictData[] = [
    {
        id: 'colombo',
        name: 'Colombo',
        coordinates: { latitude: 6.9271, longitude: 79.8612 },
        bounds: [
            [6.85, 79.80], [6.85, 79.95], [7.00, 79.95], [7.00, 79.80]
        ],
        currentStats: {
            touristCount: 12500,
            trafficLevel: 85,
            weatherTemp: 32,
            weatherCondition: 'Partly Cloudy',
            safetyScore: 92,
            crisisLevel: 'low',
            topAttractions: ['Gangaramaya Temple', 'National Museum', 'Galle Face Green']
        },
        forecast: {
            nextMonth: 15200,
            confidence: 87
        }
    },
    {
        id: 'gampaha',
        name: 'Gampaha',
        coordinates: { latitude: 7.0873, longitude: 80.0142 },
        bounds: [
            [6.95, 79.90], [6.95, 80.10], [7.15, 80.10], [7.15, 79.90]
        ],
        currentStats: {
            touristCount: 8200,
            trafficLevel: 65,
            weatherTemp: 31,
            weatherCondition: 'Sunny',
            safetyScore: 88,
            crisisLevel: 'low',
            topAttractions: ['Negombo Beach', 'Kelaniya Temple']
        },
        forecast: {
            nextMonth: 9500,
            confidence: 82
        }
    },
    {
        id: 'kalutara',
        name: 'Kalutara',
        coordinates: { latitude: 6.5854, longitude: 80.0043 },
        bounds: [
            [6.50, 79.90], [6.50, 80.10], [6.70, 80.10], [6.70, 79.90]
        ],
        currentStats: {
            touristCount: 4800,
            trafficLevel: 45,
            weatherTemp: 30,
            weatherCondition: 'Clear',
            safetyScore: 90,
            crisisLevel: 'low',
            topAttractions: ['Kalutara Beach', 'Richmond Castle']
        },
        forecast: {
            nextMonth: 5600,
            confidence: 78
        }
    },
    {
        id: 'kandy',
        name: 'Kandy',
        coordinates: { latitude: 7.2906, longitude: 80.6337 },
        bounds: [
            [7.20, 80.55], [7.20, 80.75], [7.40, 80.75], [7.40, 80.55]
        ],
        currentStats: {
            touristCount: 18500,
            trafficLevel: 72,
            weatherTemp: 28,
            weatherCondition: 'Light Rain',
            safetyScore: 94,
            crisisLevel: 'medium',
            topAttractions: ['Temple of Tooth', 'Peradeniya Gardens', 'Kandy Lake']
        },
        forecast: {
            nextMonth: 21000,
            confidence: 91
        }
    },
    {
        id: 'nuwara_eliya',
        name: 'NuwaraEliya',
        coordinates: { latitude: 6.9497, longitude: 80.7891 },
        bounds: [
            [6.85, 80.70], [6.85, 80.90], [7.05, 80.90], [7.05, 80.70]
        ],
        currentStats: {
            touristCount: 9200,
            trafficLevel: 38,
            weatherTemp: 18,
            weatherCondition: 'Foggy',
            safetyScore: 96,
            crisisLevel: 'low',
            topAttractions: ['Gregory Lake', 'Horton Plains', 'Tea Estates']
        },
        forecast: {
            nextMonth: 11500,
            confidence: 85
        }
    },
    {
        id: 'galle',
        name: 'Galle',
        coordinates: { latitude: 6.0535, longitude: 80.2210 },
        bounds: [
            [6.00, 80.15], [6.00, 80.30], [6.15, 80.30], [6.15, 80.15]
        ],
        currentStats: {
            touristCount: 14200,
            trafficLevel: 58,
            weatherTemp: 31,
            weatherCondition: 'Sunny',
            safetyScore: 91,
            crisisLevel: 'low',
            topAttractions: ['Galle Fort', 'Unawatuna Beach', 'Lighthouse']
        },
        forecast: {
            nextMonth: 16800,
            confidence: 89
        }
    },
    {
        id: 'matara',
        name: 'Matara',
        coordinates: { latitude: 5.9549, longitude: 80.5550 },
        bounds: [
            [5.92, 80.50], [5.92, 80.70], [6.10, 80.70], [6.10, 80.50]
        ],
        currentStats: {
            touristCount: 6100,
            trafficLevel: 42,
            weatherTemp: 30,
            weatherCondition: 'Clear',
            safetyScore: 87,
            crisisLevel: 'low',
            topAttractions: ['Mirissa Beach', 'Dondra Lighthouse']
        },
        forecast: {
            nextMonth: 7200,
            confidence: 80
        }
    },
    {
        id: 'hambantota',
        name: 'Hambantota',
        coordinates: { latitude: 6.1429, longitude: 81.1212 },
        bounds: [
            [6.00, 80.80], [6.00, 81.20], [6.30, 81.20], [6.30, 80.80]
        ],
        currentStats: {
            touristCount: 5400,
            trafficLevel: 35,
            weatherTemp: 33,
            weatherCondition: 'Hot & Dry',
            safetyScore: 89,
            crisisLevel: 'low',
            topAttractions: ['Yala National Park', 'Bundala', 'Tangalle Beach']
        },
        forecast: {
            nextMonth: 6800,
            confidence: 76
        }
    },
    {
        id: 'badulla',
        name: 'Badulla',
        coordinates: { latitude: 6.9934, longitude: 81.0550 },
        bounds: [
            [6.85, 80.95], [6.85, 81.20], [7.10, 81.20], [7.10, 80.95]
        ],
        currentStats: {
            touristCount: 3800,
            trafficLevel: 28,
            weatherTemp: 26,
            weatherCondition: 'Pleasant',
            safetyScore: 93,
            crisisLevel: 'low',
            topAttractions: ['Ella Rock', "Lipton's Seat", 'Nine Arch Bridge']
        },
        forecast: {
            nextMonth: 4500,
            confidence: 73
        }
    },
    {
        id: 'anuradhapura',
        name: 'Anuradhapura',
        coordinates: { latitude: 8.3114, longitude: 80.4037 },
        bounds: [
            [8.20, 80.30], [8.20, 80.50], [8.45, 80.50], [8.45, 80.30]
        ],
        currentStats: {
            touristCount: 11200,
            trafficLevel: 52,
            weatherTemp: 34,
            weatherCondition: 'Very Hot',
            safetyScore: 88,
            crisisLevel: 'medium',
            topAttractions: ['Sacred City', 'Ruwanwelisaya', 'Sri Maha Bodhi']
        },
        forecast: {
            nextMonth: 13500,
            confidence: 86
        }
    },
    {
        id: 'polonnaruwa',
        name: 'Polonnaruwa',
        coordinates: { latitude: 7.9403, longitude: 81.0188 },
        bounds: [
            [7.85, 80.95], [7.85, 81.15], [8.05, 81.15], [8.05, 80.95]
        ],
        currentStats: {
            touristCount: 7600,
            trafficLevel: 41,
            weatherTemp: 32,
            weatherCondition: 'Humid',
            safetyScore: 90,
            crisisLevel: 'low',
            topAttractions: ['Ancient City', 'Gal Vihara', 'Parakrama Samudra']
        },
        forecast: {
            nextMonth: 8900,
            confidence: 81
        }
    },
    {
        id: 'matale',
        name: 'Matale',
        coordinates: { latitude: 7.4675, longitude: 80.6234 },
        bounds: [
            [7.40, 80.50], [7.40, 80.80], [7.70, 80.80], [7.70, 80.50]
        ],
        currentStats: {
            touristCount: 4200,
            trafficLevel: 33,
            weatherTemp: 29,
            weatherCondition: 'Pleasant',
            safetyScore: 91,
            crisisLevel: 'low',
            topAttractions: ['Knuckles Range', 'Aluvihare Temple', 'Spice Gardens']
        },
        forecast: {
            nextMonth: 5100,
            confidence: 75
        }
    }
];

// Get color based on traffic level
export const getTrafficColor = (level: number): string => {
    if (level < 40) return '#4CAF50';  // Green
    if (level < 70) return '#FFC107';  // Yellow
    return '#F44336';  // Red
};

// Get color based on crisis level
export const getCrisisColor = (level: 'low' | 'medium' | 'high'): string => {
    switch (level) {
        case 'low': return '#4CAF50';
        case 'medium': return '#FF9800';
        case 'high': return '#F44336';
    }
};