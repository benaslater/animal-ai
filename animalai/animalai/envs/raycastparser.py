import enum
import numpy as np

"""
The script is designed to parse raycast observations from the AnimalAI environment. 
It includes a class to interpret these observations and output a simplified version 
of the data that only contains relevant objects. 
"""


class RayCastObjects(enum.Enum):
    """
    Enumeration of possible objects detected by the raycast. 
    The values should correspond with how they are defined in the AnimalAI Unity environment.
    """
    ARENA = 0
    IMMOVABLE = 1
    MOVABLE = 2
    GOODGOAL = 3
    GOODGOALMULTI = 4
    BADGOAL = 5
    GOALSPAWNER = 6
    DEATHZONE = 7
    HOTZONE = 8
    RAMP = 9
    PILLARBUTTON = 10
    SIGNPOSTER = 11


class RayCastParser():
    """
    The RayCastParser class is responsible for parsing raycast observations 
    from the AnimalAI environment and returning a simplified version that only 
    contains relevant objects.
    """

    def __init__(self, listOfObjects, numberOfRays):
        """
        Initialize the RayCastParser.

        Parameters:
        - listOfObjects: List of objects (from RayCastObjects enum) that the parser should look for.
        - numberOfRays: The number of rays in the raycast from AnimalAI environment.
        """
        self.numberOfRays = numberOfRays
        self.listOfObjects = listOfObjects
        self.listofObjectVals = [x.value for x in listOfObjects]
        self.numberOfObjects = len(listOfObjects)

    def parse(self, raycast) -> np.ndarray:
        """
        Parse the raw raycast data and return a simplified array.

        Parameters:
        - raycast: The raw raycast array from the AnimalAI environment.

        Returns:
        - np.ndarray: The parsed raycast, simplified to only include objects in listOfObjects.
        """
        if isinstance(raycast, dict):
            raycast = raycast['rays']

        self.numberDetectableObjects = int(
            len(raycast) / self.numberOfRays) - 2
        parsedRaycast = np.zeros((len(self.listOfObjects), self.numberOfRays))
        for i in range(self.numberOfRays):
            for j in range(self.numberDetectableObjects):
                idx = i * (self.numberDetectableObjects + 2) + j
                distance_idx = i * \
                    (self.numberDetectableObjects + 2) + \
                    self.numberDetectableObjects + 1
                if idx < len(raycast) and j in self.listofObjectVals:
                    if raycast[idx] == 1 and distance_idx < len(raycast):
                        parsedRaycast[self.listofObjectVals.index(
                            j)][i] = raycast[distance_idx]

        parsedRaycast = np.reshape(
            parsedRaycast, (len(self.listOfObjects), self.numberOfRays))
        reordered = np.zeros_like(parsedRaycast)
        for i in range(parsedRaycast.shape[0]):
            reordered[i] = self.reorderRow(parsedRaycast[i])
        return reordered

    def reorderRow(self, row):
        """
        Reorder the elements in a row so that they read from left to right, 
        as opposed to from the middle outwards.

        Parameters:
        - row: A 1D numpy array representing a single row of parsed raycast data.

        Returns:
        - np.ndarray: The reordered row.
        """
        newRow = np.zeros_like(row)
        midIndex = int((self.numberOfRays-1)/2)
        newRow[midIndex] = row[0]
        for i in range(midIndex):
            newRow[i+1+midIndex] = row[self.numberOfRays-2*(i+1)]
            newRow[i] = row[2*(i+1)]
        return newRow

    def prettyPrint(self, raycast) -> str:
        """
        Pretty-prints the parsed raycast data, making it easier to read and understand.

        Parameters:
        - raycast: The raw raycast array from the AnimalAI environment.

        Prints the parsed and simplified raycast in a human-readable format.
        """
        if isinstance(raycast, dict):
            raycast = raycast['rays']

        parsedRaycast = self.parse(raycast)
        for i in range(parsedRaycast.shape[0]):
            print(self.listOfObjects[i].name, ":", parsedRaycast[i])


if __name__ == "__main__":
    tests = [
        [
            # Test 1: Simple test to check if GOODGOAL and IMMOVABLE objects are correctly parsed
            # Description: Checks if the parser correctly identifies GOODGOAL and IMMOVABLE objects
            # and places them correctly in the parsed raycast array.
            [RayCastObjects.GOODGOAL, RayCastObjects.IMMOVABLE],
            5,
            [1, 1, 1, 1, 1, 1, 0, 0.1,
            1, 1, 1, 1, 1, 1, 0, 0.2,
            1, 1, 1, 1, 1, 1, 1, 0.3,
            1, 1, 1, 1, 1, 1, 1, 0.4,
            1, 1, 1, 1, 1, 1, 1, 0.5],
            [[0.3, 0.5, 0.1, 0.4, 0.2], [0.3, 0.5, 0.1, 0.4, 0.2]]
        ],
        [
            # Test 2: Checking for no objects detected
            # Description: Checks if the parser correctly identifies when no objects are detected.
            [RayCastObjects.GOODGOAL, RayCastObjects.IMMOVABLE],
            3,
            [0, 0, 0, 0, 0, 0, 0, 0.1,
            0, 0, 0, 0, 0, 0, 0, 0.2,
            0, 0, 0, 0, 0, 0, 0, 0.3],
            [[0, 0, 0], [0, 0, 0]]
        ],
        [
            # Test 3: Mix of objects detected and not detected
            # Description: Checks the parser correctly identifies some objects and ignores others.
            [RayCastObjects.ARENA, RayCastObjects.MOVABLE, RayCastObjects.GOODGOALMULTI],
            7,
            [1, 0, 0, 0, 0, 0, 0, 0.1,
            0, 1, 0, 0, 0, 0, 0, 0.2,
            0, 0, 1, 0, 0, 0, 1, 0.3,
            0, 0, 0, 1, 0, 0, 1, 0.4,
            0, 0, 0, 0, 1, 0, 1, 0.5,
            0, 0, 0, 0, 0, 1, 1, 0.6,
            0, 0, 0, 0, 0, 0, 0, 0],
            [[0,  0,  0,  0.1, 0,  0,  0],
            [0.3, 0,  0,  0,  0,  0,  0, ],
            [0,  0.5, 0,  0,  0,  0,  0 ]]
        ],
        [
            # Test 4: Mix of objects detected and not detected, including PILLARBUTTON
            # Description: Check for object identification/ignoring, including PILLARBUTTON
            [RayCastObjects.ARENA, RayCastObjects.PILLARBUTTON, RayCastObjects.MOVABLE],
            7,
            [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0.1,
            0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0.2,
            0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0.3,
            0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0.4,
            0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0.5,
            0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0.6,
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
            [[0,  0,  0,  0.1, 0,  0,  0, ],
            [0.3, 0.5, 0,  0.1, 0.6, 0.4, 0.2],
            [0.3, 0,  0,  0,  0,  0,  0 ]]
        ]
    ]

    # TODO: Transfer tests to a testing framework
    # TODO: Add assertations for pretty printing
    for i, [parser_objects, no_rays, test_raycast, expected_parsed_raycast] in enumerate(tests):
        rayParser = RayCastParser(
            parser_objects, no_rays)
        parsedRaycast = rayParser.parse(test_raycast)
        assert np.array_equal(
            parsedRaycast,
            expected_parsed_raycast
        ), f"Failed comparison of parsed and expected raycast for Test {i}"
        print(f"Parsed Raycast for Test {i}:")
        print(parsedRaycast)
        rayParser.prettyPrint(test_raycast)
